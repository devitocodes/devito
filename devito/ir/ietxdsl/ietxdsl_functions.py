# definitions pulled out from GenerateXDSL jupyter notebook
import ctypes
import numpy
from sympy import Indexed, Integer, Symbol, Add, Eq, Mod, Pow, Mul, Float
import cgen

from typing import Any

import devito.ir.iet.nodes as nodes

from devito import SpaceDimension
from devito.passes.iet.languages.openmp import OmpRegion

from devito.ir.ietxdsl import (MLContext, IET, Constant, Modi, Block, Statement,
                               PointerCast, Powi, Initialise, StructDecl, Call)
from devito.tools import as_list
from devito.tools.utils import as_tuple
from devito.types.basic import IndexedData

# XDSL specific imports
from xdsl.irdl import AnyOf, Operation, SSAValue
from xdsl.dialects.builtin import (ContainerOf, Float16Type, Float32Type,
                                   Float64Type, i32, f32)

from devito.ir.ietxdsl import iet_ssa

from xdsl.dialects import memref, arith, builtin
from xdsl.dialects.experimental import math

import devito.types

floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))

def printHeaders(cgen, header_str, headers):
    for header in headers:
        cgen.printOperation(Statement.get(createStatement(header_str, header)))
    cgen.printOperation(Statement.get(createStatement('')))


def printIncludes(cgen, header_str, headers):
    for header in headers:
        cgen.printOperation(Statement.get(
                            createStatement(header_str, '"' + header + '"')))
    cgen.printOperation(Statement.get(createStatement('')))


def printStructs(cgen, struct_decs):
    for struct in struct_decs:
        cgen.printOperation(
            StructDecl.get(struct.tpname, struct.fields, struct.declname,
                           struct.pad_bytes))


def print_calls(cgen, calldefs):

    for node in calldefs:
        call_name = str(node.root.name)

        """
        (Pdb) calldefs[0].root.args['parameters']
        [buf(x), x_size, f(t, x), otime, ox]
        (Pdb) calldefs[0].root.args['parameters'][0]
        buf(x)
        (Pdb) calldefs[0].root.args['parameters'][0]._C_name
        """
        try:
            C_names = [str(i._C_name) for i in node.root.args['parameters']]
            C_typenames = [str(i._C_typename) for i in node.root.args['parameters']]
            C_typeqs = [str(i._C_type_qualifier) for i in node.root.args['parameters']]
            prefix = node.root.prefix[0]
            retval = node.root.retval
        except:
            print("Call not translated in calldefs")
            return

        call = Call(call_name, C_names, C_typenames, C_typeqs, prefix, retval)

        cgen.printCall(call, True)


def createStatement(string="", val=None):
    for t in as_tuple(val):
        string = string + " " + t

    return string


def collectStructs(parameters):
    struct_decs = []
    struct_strs = []
    for i in parameters:
        # Bypass a struct decl if it has te same _C_typename
        if (i._C_typedecl is not None and str(i._C_typename) not in struct_strs):
            struct_decs.append(i._C_typedecl)
            struct_strs.append(i._C_typename)
    return struct_decs


def calculateAddArguments(arguments):
    # Get an input of arguments that are added. In case only one argument remains,
    # return the argument.
    # In case more, return expression by breaking down args.
    if len(arguments) == 1:
        return arguments[0]
    else:
        return Add(arguments[0], calculateAddArguments(arguments[1:len(arguments)]))


def add_to_block(expr, arg_by_expr: dict[Any, Operation], result):
    # # import pdb;pdb.set_trace()
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
        arg = Constant.from_int_and_width(constant, i32)
        arg_by_expr[expr] = arg
        result.append(arg)
        return

    if isinstance(expr, Float):
        constant = float(expr.evalf())
        arg = Constant.from_float_and_width(constant, f32)
        arg_by_expr[expr] = arg
        result.append(arg)
        return

    for child_expr in expr.args:
        add_to_block(child_expr, arg_by_expr, result)

    if isinstance(expr, Add):
        # workaround for large additions (TODO: should this be handled earlier?)
        len_args = len(expr.args)
        if len_args > 2:
            # this works for 3 arguments:
            first_arg = expr.args[0]
            second_arg = calculateAddArguments(expr.args[1:len_args])
            add_to_block(second_arg, arg_by_expr, result)
        else:
            first_arg = expr.args[0]
            second_arg = expr.args[1]
            # Mostly additions in indexing
            if isinstance(second_arg, SpaceDimension) | isinstance(second_arg, Indexed):
                tmp = first_arg
                first_arg = second_arg
                second_arg = tmp
        lhs = arg_by_expr[first_arg]
        rhs = arg_by_expr[second_arg]
        if isinstance(SSAValue.get(lhs).typ, builtin.IntegerType):
            sum = arith.Addi.get(lhs, rhs)
        else:
            sum = arith.Addf.get(lhs, rhs)

        arg_by_expr[expr] = sum
        result.append(sum)
        return

    if isinstance(expr, Mul):
        # Convert a sympy.core.mul.Mul to xdsl.dialects.arith.Muli
        lhs = SSAValue.get(arg_by_expr[expr.args[0]])
        rhs = SSAValue.get(arg_by_expr[expr.args[1]])

        if rhs.typ != lhs.typ:
            # reconcile differences

            if isinstance(rhs.typ, builtin.IntegerType):
                rhs = arith.SIToFPOp(rhs, lhs.typ)
                result.append(rhs)
            else:
                lhs = arith.SIToFPOp(lhs, rhs.typ)
                result.append(lhs)

        
        # happy path
        if isinstance(rhs.typ, builtin.IntegerType):
            mul = arith.Muli.get(lhs, rhs)
        else:
            mul = arith.Mulf.get(lhs, rhs)
        
        arg_by_expr[expr] = mul
        result.append(mul)
        return

    if isinstance(expr, nodes.Return):
        # Covert a Return node
        # # import pdb;pdb.set_trace()
        return

    if isinstance(expr, Mod):
        # To update docstring
        lhs = arg_by_expr[expr.args[0]]
        rhs = arg_by_expr[expr.args[1]]
        sum = arith.RemSI.get(lhs, rhs)
        arg_by_expr[expr] = sum
        result.append(sum)
        return

    if isinstance(expr, Pow):
        # Convert sympy.core.power.Pow to devito.ir.ietxdsl.operations.Powi
        base = arg_by_expr[expr.args[0]]
        exponent = arg_by_expr[expr.args[1]]
        pow = math.FPowIOp.get(base, exponent)
        arg_by_expr[expr] = pow
        result.append(pow)
        return

    if isinstance(expr, Indexed):
        # import pdb;pdb.set_trace()
        for arg in expr.args:
            add_to_block(arg, arg_by_expr, result)

        indices_list = as_list(arg_by_expr[i] for i in expr.indices)
        idx = memref.Load.get(arg_by_expr[expr.args[0]], indices_list)
        result.append(idx)
        arg_by_expr[expr] = idx
        return

    if isinstance(expr, Eq):
        # Convert devito.ir.equations.equation.ClusterizedEq to devito.ir.ietxdsl.operations.Assign

        add_to_block(expr.args[0], arg_by_expr, result)
        # lhs = arg_by_expr[expr.args[0]]
        add_to_block(expr.args[1], arg_by_expr, result)
        # rhs = arg_by_expr[expr.args[1]]

        indices_list = as_list(arg_by_expr[i] for i in expr.args[0].indices)
        load: memref.Load = arg_by_expr[expr.args[0]]
        assign = memref.Store.get(arg_by_expr[expr.args[1]], load.memref, as_list(load.indices))

        # assign = memref.Store.get(rhs, lhs)
        # assign = Assign.build([lhs, rhs])
        result.append(assign)
        arg_by_expr[expr] = assign

        return

    assert False, f'unsupported expr {expr} of type {expr.func}'


def myVisit(node, block: Block, ssa_vals={}):
    try:
        bool_node = isinstance(
            node, nodes.Node), f'Argument must be subclass of Node, found: {node}'
        comment_node = isinstance(
            node, cgen.Comment), f'Argument must be subclass of Node, found: {node}'
        statement_node = isinstance(
            node, cgen.Statement), f'Argument must be subclass of Node, found: {node}'
        assert bool_node or comment_node or statement_node
    except:
        print("fail!")

    if hasattr(node, 'is_Callable') and node.is_Callable:
        return

    if isinstance(node, nodes.CallableBody):
        return

    if isinstance(node, nodes.Expression):
        b = Block([i32])
        r = []
        expr = node.expr
        if node.init:
            expr_name = expr.args[0]
            add_to_block(expr.args[1], {Symbol(s): a for s, a in ssa_vals.items()}, r)

            # init = Initialise.get(r[-1].results[0], r[-1].results[0], str(expr_name))
            block.add_ops(r)
            ssa_vals[str(expr_name)] = r[-1].results[0]
        else:
            add_to_block(expr, {Symbol(s): a for s, a in ssa_vals.items()}, r)
            block.add_ops(r)
        return


    if isinstance(node, nodes.ExpressionBundle):
        assert len(node.children) == 1
        for idx in range(len(node.children[0])):
            child = node.children[0][idx]
            myVisit(child, block, ssa_vals)
        return

    if isinstance(node, nodes.Iteration):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1

        # Get index variable
        dim = node.dim
        assert len(node.limits) == 3, "limits should be a (min, max, step) tuple!"

        start, end, step = node.limits
        try:
            step = int(step)
        except:
            raise ValueError("step must be int!")

        # get start, end ssa values
        start_ssa_val = ssa_vals[start.name]
        end_ssa_val = ssa_vals[end.name]
        
        step_op = arith.Constant.from_int_and_width(step, i32)

        block.add_op(step_op)

        props = [str(x) for x in node.properties]
        pragmas = [str(x) for x in node.pragmas]

        subindices = len(node.uindices)

        # construct iet for operation
        loop = iet_ssa.For.get(start_ssa_val, end_ssa_val, step_op, subindices, props, pragmas)

        # extend context to include loop index
        ssa_vals[node.index] = loop.block.args[0]
        
        # TODO: add subindices to ctx
        for i, uindex in enumerate(node.uindices):
            ssa_vals[uindex.name] = loop.block.args[i+1]

        # visit the iteration body, adding ops to the loop body
        myVisit(node.children[0][0], loop.block, ssa_vals)

        # add loop to program
        block.add_op(loop)
        return

    if isinstance(node, nodes.Section):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1
        for content in node.ccode.contents:
            if isinstance(content, cgen.Comment):
                comment = Statement.get(content)
                block.add_ops([comment])
            else:
                myVisit(node.children[0][0], block, ssa_vals)
        return

    if isinstance(node, nodes.HaloSpot):
        assert len(node.children) == 1
        try:
            assert isinstance(node.children[0], nodes.Iteration)
        except:
            assert isinstance(node.children[0], OmpRegion)

        myVisit(node.children[0], block, ssa_vals)
        return

    if isinstance(node, nodes.TimedList):
        assert len(node.children) == 1
        assert len(node.children[0]) == 1
        header = Statement.get(node.header[0])
        block.add_ops([header])
        myVisit(node.children[0][0], block, ssa_vals)
        footer = Statement.get(node.footer[0])
        block.add_ops([footer])
        return

    if isinstance(node, nodes.PointerCast):
        statement = node.ccode

        assert node.defines[0]._C_name == node.obj._C_name, "This should not happen"

        # We want to know the dimensions of the u_vec->data result
        # we assume that the result will always be of dim:
        # (u_vec->size[i]) for some i
        # we further assume, that node.function.symbolic_shape
        # is always (u_vec->size[0], u_vec->size[1], ... ,u_vec->size[rank])  
        # this means that this pretty hacky way works to get the indices of the dims
        # in `u_vec->size`
        shape = (node.function.symbolic_shape.index(shape) for shape in node.castshape)

        arg = ssa_vals[node.function._C_name]
        pointer_cast = PointerCast.get(
            arg,
            statement,
            shape,
            memref_type_from_indexed_data(node.obj)
        )
        block.add_ops([pointer_cast])
        ssa_vals[node.obj._C_name] = pointer_cast.result
        return

    if isinstance(node, nodes.List):
        # Problem: When a List is ecountered with only body, but no header or footer
        # we have a problem
        for h in node.header:
            myVisit(h, block, ssa_vals)

        for b in node.body:
            myVisit(b, block, ssa_vals)

        for f in node.footer:
            myVisit(f, block, ssa_vals)

        return

    if isinstance(node, nodes.Call):
        # Those parameters without associated types aren't printed in the Kernel header
        call_name = str(node.name)

        try:
            C_names = [str(i._C_name) for i in node.arguments]
            C_typenames = [str(i._C_typename) for i in node.arguments]
            C_typeqs = [str(i._C_type_qualifier) for i in node.arguments]
            prefix = ''
            retval = ''
        except:
            # Needs to be fixed
            comment = Statement.get(node)
            block.add_ops([comment])
            print(f"Call {node.name} instance translated as comment")
            return

        call = Call(call_name, C_names, C_typenames, C_typeqs, prefix, retval)
        block.add_ops([call])

        print(f"Call {node.name} translated")
        return

    if isinstance(node, nodes.Conditional):
        # Those parameters without associated types aren't printed in the Kernel header
        print("Conditional placement skipping")
        return

    if isinstance(node, nodes.Definition):
        print("Translating definition")
        comment = Statement.get(node)
        block.add_ops([comment])
        return

    if isinstance(node, cgen.Comment):
        # cgen.Comment as Statement
        comment = Statement.get(node)
        block.add_ops([comment])
        return

    if isinstance(node, cgen.Statement):
        comment = Statement.get(node)
        block.add_ops([comment])
        return

    if isinstance(node, cgen.Line):
        comment = Statement.get(node)
        block.add_ops([comment])
        return

    #raise TypeError(f'Unsupported type of node: {type(node)}, {vars(node)}')


def get_arg_types(symbols):
    processed = []
    for symbol in symbols:
        if isinstance(symbol, IndexedData):
            processed.append(
                memref_type_from_indexed_data(symbol)
            )
        elif isinstance(symbol, devito.types.misc.Timer):
            processed.append(
                iet_ssa.Profiler()
            )
        elif symbol._C_typedata == 'struct dataobj':
            processed.append(
                iet_ssa.Dataobj()
            )
        elif symbol._C_ctype == ctypes.c_float:
            processed.append(f32)
        else:
            assert symbol._C_ctype == ctypes.c_int
            # TODO: inspect symbol._C_ctype to gather type info
            processed.append(i32)

    return processed


def memref_type_from_indexed_data(d: IndexedData):
    stype = dtypes_to_xdsltypes[d.dtype]
    return memref.MemRefType.from_element_type_and_shape(stype, [-1]*len(d.shape))


dtypes_to_xdsltypes = {
    numpy.float32: builtin.f32,
    numpy.float64: builtin.f64,
    numpy.int32: builtin.i32,
    numpy.int64: builtin.i64,
}
