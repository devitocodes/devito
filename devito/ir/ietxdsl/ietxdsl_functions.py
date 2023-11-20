# definitions pulled out from GenerateXDSL jupyter notebook
import ctypes
import numpy
from sympy import Indexed, Integer, Symbol, Add, Eq, Mod, Pow, Mul, Float

from typing import Any

import devito.ir.iet.nodes as nodes

from devito import SpaceDimension

from devito.tools import as_list
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
        arg = arith.Constant.from_int_and_width(constant, i32)
        arg_by_expr[expr] = arg
        result.append(arg)
        return

    if isinstance(expr, Float):
        constant = float(expr.evalf())
        arg = arith.Constant.from_float_and_width(constant, f32)
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
        # Convert devito.ir.equations.equation.ClusterizedEq to
        # devito.ir.ietxdsl.operations.Assign

        add_to_block(expr.args[0], arg_by_expr, result)
        # lhs = arg_by_expr[expr.args[0]]
        add_to_block(expr.args[1], arg_by_expr, result)
        # rhs = arg_by_expr[expr.args[1]]

        indices_list = as_list(arg_by_expr[i] for i in expr.args[0].indices)
        load: memref.Load = arg_by_expr[expr.args[0]]
        assign = memref.Store.get(arg_by_expr[expr.args[1]],
                                  load.memref,
                                  as_list(load.indices))

        # assign = memref.Store.get(rhs, lhs)
        # assign = Assign.build([lhs, rhs])
        result.append(assign)
        arg_by_expr[expr] = assign

        return

    assert False, f'unsupported expr {expr} of type {expr.func}'


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
