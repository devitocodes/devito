import numpy as np
import itertools

from sympy import sympify
from sympy.core.numbers import Zero

from devito import Eq
from devito.ir.equations import ClusterizedEq
from devito.ir.iet.nodes import Call, Callable, Expression, IterationTree
from devito.ir.iet.visitors import FindNodes, FindSymbols
from devito.ops.node_factory import OPSNodeFactory
from devito.ops.types import Array, OpsAccessible, OpsDat, OpsStencil
from devito.ops.utils import namespace
from devito.symbolics import Byref, ListInitializer, Literal
from devito.types import DefaultDimension, Symbol
from devito.tools import filter_sorted


def opsit(trees, count, block):
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

    it_range = []
    for tree in trees:
        expressions.extend(FindNodes(Expression).visit(tree.inner))
        if isinstance(tree, IterationTree):
            it_range = [it.bounds() for it in tree]

    par_loop_array = Array(
        name=namespace['ops_par_loop_range_name'],
        dimensions=(DefaultDimension(name='len', default_value=sum(len(x)
                                                                   for x in it_range)),),
        dtype=np.int32,
    )

    par_loop_range = Expression(ClusterizedEq(Eq(
        Symbol(par_loop_array.name),
        ListInitializer(list(itertools.chain(*it_range)))
    )))

    for expr in expressions:
        ops_expressions.append(Expression(make_ops_ast(expr.expr, node_factory)))

    parameters = sorted(node_factory.ops_params, key=lambda i: (i.is_Constant, i.name))

    ops_kernel = Callable(
        namespace['ops_kernel'](count),
        ops_expressions,
        "void",
        parameters
    )

    # Extract all symbols that need to be converted to ops_dat
    to_dat = set()
    symbols = set(FindSymbols('symbolics').visit(trees[0].root))
    symbols -= set(FindSymbols('defines').visit(trees[0].root))
    to_dat |= symbols

    # To ensure deterministic code generation we order the datasets to
    # be generated (since a set is an unordered collection)
    to_dat = filter_sorted(to_dat)

    name_to_ops_dat = {}
    pre_loop = []
    for f in to_dat:
        if f.is_Constant:
            continue

        _ops_dat = create_ops_dat(f, name_to_ops_dat, block)
        pre_loop.extend(_ops_dat)

    name_to_ops_stencil = {}
    stencil_arrays_initializations = [
        to_ops_stencil(p, node_factory.ops_args_accesses[p], name_to_ops_stencil)
        for p in parameters if isinstance(p, OpsAccessible)
    ]

    ops_args = []
    for p in parameters:
        if not isinstance(p, OpsAccessible):
            ops_args.append(Call(namespace['ops_arg_gbl'],
                                 [Byref(Literal(p.name.replace('*', ''))),
                                  1, Literal('"%s"' % p._C_typedata),
                                  Literal(namespace['ops_read'])]))
        else:
            ops_args.append(Call(namespace['ops_arg_dat'],
                                 [Literal('%s' % name_to_ops_dat[p.name]), 1,
                                  Literal(name_to_ops_stencil[p.name]),
                                  Literal('"float"'),
                                  Literal('%s' % namespace['ops_read']
                                          if p.read_only else namespace['ops_write'])]))

    par_loop = Call(namespace['ops_par_loop'], [
        Literal(ops_kernel.name),
        Literal('"%s"' % ops_kernel.name),
        Symbol(block.name),
        2,
        Symbol(par_loop_array.name),
        *ops_args
    ])
    pre_time_loop = itertools.chain(
        *[pre_loop], *[stencil_arrays_initializations], [par_loop_range])

    return pre_time_loop, ops_kernel


def to_ops_stencil(param, accesses, name_to_ops_stencil):
    dims = len(accesses[0])
    pts = len(accesses)
    stencil_name = namespace['ops_stencil_name'](dims, param.name, pts)

    stencil_array = Array(
        name=stencil_name,
        dimensions=(DefaultDimension(name='len', default_value=dims * pts),),
        dtype=np.int32,
    )

    ops_stencil = OpsStencil(stencil_name.upper())
    name_to_ops_stencil[param.name] = ops_stencil.name

    return [
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
        dtype=np.int32
    )
    base = Array(
        name=namespace['ops_dat_base'](f.name),
        dimensions=(DefaultDimension(name='base', default_value=ndim),),
        dtype=np.int32
    )
    d_p = Array(
        name=namespace['ops_dat_d_p'](f.name),
        dimensions=(DefaultDimension(name='d_p', default_value=ndim),),
        dtype=np.int32
    )
    d_m = Array(
        name=namespace['ops_dat_d_m'](f.name),
        dimensions=(DefaultDimension(name='d_m', default_value=ndim),),
        dtype=np.int32
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
        d_m_val = tuple(sympify([-(p[1] + h[1]) for p, h in zip(padding, halo)]))

        ops_dat_array = Array(
            name=namespace['ops_dat_name'](f.name),
            dimensions=(DefaultDimension(name='dat', default_value=time_dims),),
            dtype='ops_dat',
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

        d_p_val = tuple(sympify([p[0] + h[0] for p, h in zip(f.padding, f.halo)]))
        d_m_val = tuple(sympify([-(p[1] + h[1]) for p, h in zip(f.padding, f.halo)]))
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
