# definitions pulled out from GenerateXDSL jupyter notebook
from sympy import Indexed, Integer, Symbol, Add, Eq, Mod, Pow, Mul
from cgen import Generable

from devito.tools import flatten
from devito.ir import retrieve_iteration_tree
from devito.ir.ietxdsl import (MLContext, Builtin, IET, Constant, Addi, Modi, Idx,
                               Assign, Block, Iteration, IterationWithSubIndices,
                               Statement, PointerCast, Powi, Initialise, Muli,
                               StructDecl)
from devito import ModuloDimension
import devito.ir.iet.nodes as nodes
from devito.types.basic import IndexedData

ctx = MLContext()
Builtin(ctx)
iet = IET(ctx)


def printHeaders(cgen, header_str, headers):
    for header in headers:
        cgen.printOperation(Statement.get(createStatement(header_str, header)))


def printStructs(cgen, struct_decs):
    for struct in struct_decs:
        cgen.printOperation(
            StructDecl.get(struct.tpname, struct.fields, struct.declname,
                           struct.pad_bytes))


def getOpParamsNames(op, op_param_names):
    body = op.body.body[1].args.get('body')
    body_size = len(body)
    # then add in anything that comes before the main loop:
    for i in range(0, (body_size - 1)):
        val = op.body.body[1].args.get('body')[0]._args['body'][0].write
        op_param_names.append(str(val))

    # we still need to add the extra time indices even though they aren't passed in
    devito_iterations = flatten(retrieve_iteration_tree(op.body))
    timing_indices = [i.uindices for i in devito_iterations if i.dim.is_Time]
    for tup in timing_indices:
        for t in tup:
            op_param_names.append((str(t)))
    return op_param_names


def createStatement(initial_string, val):
    ret_str = initial_string
    if isinstance(val, tuple):
        for t in val:
            ret_str = ret_str + " " + t
    else:
        ret_str = ret_str + " " + val

    return ret_str


def collectStructs(parameters):
    struct_decs = []
    for i in parameters:
        if (i._C_typedecl is not None and i._C_typedecl not in struct_decs):
            struct_decs.append(i._C_typedecl)
    return struct_decs


def add_to_block(expr, arg_by_expr, result):
    if expr in arg_by_expr:
        return

    if isinstance(expr, IndexedData):
        # Only index first bit of IndexedData
        add_to_block(expr.args[0], arg_by_expr, result)
        arg_by_expr[expr] = arg_by_expr[expr.args[0]]
        return

    if isinstance(expr, Symbol):
        # All symbols must be passed in at the start
        my_expr = Symbol(expr.name)
        assert my_expr in arg_by_expr, f'Symbol with name {expr.name} not found ' \
                                       f'in {arg_by_expr}'
        arg_by_expr[expr] = arg_by_expr[my_expr]
        return

    if isinstance(expr, Integer):
        constant = int(expr.evalf())
        arg = Constant.get(constant)
        arg_by_expr[expr] = arg
        result.append(arg)
        return

    for child_expr in expr.args:
        add_to_block(child_expr, arg_by_expr, result)

    if isinstance(expr, Add):
        lhs = arg_by_expr[expr.args[0]]
        rhs = arg_by_expr[expr.args[1]]
        sum = Addi.get(lhs, rhs)
        arg_by_expr[expr] = sum
        result.append(sum)
        return

    if isinstance(expr, Mul):
        lhs = arg_by_expr[expr.args[0]]
        rhs = arg_by_expr[expr.args[1]]
        sum = Muli.get(lhs, rhs)
        arg_by_expr[expr] = sum
        result.append(sum)
        return

    if isinstance(expr, Mod):
        lhs = arg_by_expr[expr.args[0]]
        rhs = arg_by_expr[expr.args[1]]
        sum = Modi.get(lhs, rhs)
        arg_by_expr[expr] = sum
        result.append(sum)
        return

    if isinstance(expr, Pow):
        base = arg_by_expr[expr.args[0]]
        exponent = arg_by_expr[expr.args[1]]
        pow = Powi.get(base, exponent)
        arg_by_expr[expr] = pow
        result.append(pow)
        return

    if isinstance(expr, Indexed):
        add_to_block(expr.args[0], arg_by_expr, result)
        prev = arg_by_expr[expr.args[0]]
        for child_expr in expr.args[1:]:
            add_to_block(child_expr, arg_by_expr, result)
            child_arg = arg_by_expr[child_expr]
            idx = Idx.get(prev, child_arg)
            result.append(idx)
            prev = idx
        arg_by_expr[expr] = prev
        return

    if isinstance(expr, Eq):
        add_to_block(expr.args[0], arg_by_expr, result)
        lhs = arg_by_expr[expr.args[0]]
        add_to_block(expr.args[1], arg_by_expr, result)
        rhs = arg_by_expr[expr.args[1]]
        assign = Assign.build([lhs, rhs])
        arg_by_expr[expr] = assign
        result.append(assign)
        return

    assert False, f'unsupported expr {expr} of type {expr.func}'


def myVisit(node, block=None, ctx={}):
    assert isinstance(
        node, nodes.Node), f'Argument must be subclass of Node, found: {node}'

    if hasattr(node, 'is_Callable') and node.is_Callable:
        return

    if isinstance(node, nodes.CallableBody):
        return

    if isinstance(node, nodes.Expression):
        expr = node.expr
        b = Block.from_arg_types([iet.i32])
        r = []
        if node.init:
            expr_name = expr.args[0]
            add_to_block(expr.args[1], {Symbol(s): a for s, a in ctx.items()}, r)
            init = Initialise.get(r[-1].results[0], [iet.f32], str(expr_name))
            block.add_ops([init])
        else:
            add_to_block(expr, {Symbol(s): a for s, a in ctx.items()}, r)
            block.add_ops(r)
        return

    if isinstance(node, nodes.ExpressionBundle):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1
        myVisit(node.children[0][0], block, ctx)
        return

    if isinstance(node, nodes.Iteration):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1
        index = node.index
        b = Block.from_arg_types([iet.i32])
        ctx = {**ctx, index: b.args[0]}
        # check if there are subindices
        hasSubIndices = False
        if len(node.uindices) > 0:
            uindices_names = []
            uindices_symbmins = []
            for uindex in list(node.uindices):
                # currently only want to deal with a very specific subindex!
                if isinstance(uindex, ModuloDimension):
                    hasSubIndices = True
                    uindices_names.append(uindex.name)
                    uindices_symbmins.append(uindex.symbolic_min)
            if hasSubIndices:
                myVisit(node.children[0][0], b, ctx)
                if len(node.pragmas) > 0:
                    for p in node.pragmas:
                        prag = Statement.get(p)
                        block.add_ops([prag])
                iteration = IterationWithSubIndices.get(
                    node.properties, node.limits, uindices_names,
                    uindices_symbmins, node.index, b)
                block.add_ops([iteration])
                return

        myVisit(node.children[0][0], b, ctx)
        if len(node.pragmas) > 0:
            for p in node.pragmas:
                prag = Statement.get(p)
                block.add_ops([prag])
        iteration = Iteration.get(node.properties, node.limits, node.index, b)
        block.add_ops([iteration])
        return

    if isinstance(node, nodes.Section):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1
        # TODO: there doesn't seem to be a straightforward way of pulling out the
        # necessary parts of a Section..
        for content in node.ccode.contents:
            if isinstance(content, Generable):
                comment = Statement.get(content)
                block.add_ops([comment])
            elif isinstance(content, node.Collection):
                myVisit(node.children[0][0], block, ctx)
        return

    if isinstance(node, nodes.TimedList):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1
        header = Statement.get(node.header[0])
        block.add_ops([header])
        myVisit(node.children[0][0], block, ctx)
        footer = Statement.get(node.footer[0])
        block.add_ops([footer])
        return

    if isinstance(node, nodes.PointerCast):
        statement = node.ccode
        pointer_cast = PointerCast.get(statement)
        block.add_ops([pointer_cast])
        return

    if isinstance(node, nodes.List):
        header = node.header
        for h in header:
            comment = Statement.get(h)
            block.add_ops([comment])
        footer = node.footer
        for h in footer:
            comment = Statement.get(h)
            block.add_ops([comment])
        return

    raise TypeError(f'Unsupported type of node: {type(node)}, {vars(node)}')
