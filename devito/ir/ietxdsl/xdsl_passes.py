from devito import Operator

from devito.ir import PointerCast, FindNodes
from devito.ir.iet import FindSymbols
from devito.ir.iet.nodes import CallableBody, MetaCall, Definition, Dereference, Prodder  # noqa

from devito.ir.ietxdsl import (MLContext, IET, CGeneration,
                               ietxdsl_functions, Callable)

from devito.ir.ietxdsl.ietxdsl_functions import collectStructs, get_arg_types
from xdsl.dialects.builtin import Builtin, i32
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region


def transform_devito_xdsl_string(op: Operator):

    """
    Transform a Devito Operator to an XDSL code string.
    Parameters
    ----------
    op : Operator
        A Devito Operator.
    Returns
    -------
    cgen.str
        A cgen string with the transformed code.
    """

    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    cgen = CGeneration()

    # Print headers/includes/Structs
    ietxdsl_functions.printHeaders(cgen, "#define", op._headers)
    ietxdsl_functions.printIncludes(cgen, "#include", op._includes)
    ietxdsl_functions.printStructs(cgen, collectStructs(op.parameters))

    # Check for the existence of funcs in the operator (print devito metacalls)
    op_funcs = [value for _, value in op._func_table.items()]
    # Print calls
    ietxdsl_functions.print_calls(cgen, op_funcs)
    # Visit and print the main kernel
    call_obj = _op_to_func(op)
    cgen.printCallable(call_obj)

    # After finishing kernels, now we check the rest of the functions
    module = builtin.ModuleOp.from_region_or_ops([call_obj])
    for op_func in op_funcs:
        op = op_func.root
        name = op.name

        # Those parameters without associated types aren't printed in the Kernel header
        op_param_names = [s._C_name for s in FindSymbols('defines').visit(op)]
        op_header_params = [i._C_name for i in list(op.parameters)]
        op_types = [i._C_typename for i in list(op.parameters)]
        op_type_qs = [i._C_type_qualifier for i in list(op.parameters)]
        prefix = '-'.join(op.prefix)
        retval = str(op.retval)
        # import pdb;pdb.set_trace()
        b = Block([i32] * len(op_param_names))
        d = {name: register for name, register in zip(op_param_names, b.args)}

        # Add Allocs
        for op_alloc in op.body.allocs:
            ietxdsl_functions.myVisit(op_alloc, block=b, ssa_vals=d)

        cgen.print('')
        # Add obj defs
        for op_obj in op.body.objs:
            ietxdsl_functions.myVisit(op_obj, block=b, ssa_vals=d)

        # import pdb;pdb.set_trace()

        # Add Casts
        for cast in FindNodes(PointerCast).visit(op.body):
            ietxdsl_functions.myVisit(cast, block=b, ssa_vals=d)

        call_obj = Callable.get(name, op_param_names, op_header_params, op_types,
                                op_type_qs, retval, prefix, b)

        for body_i in op.body.body:
            # Comments
            if body_i.args.get('body') != ():
                for body_j in body_i.body:
                    # Casts
                    ietxdsl_functions.myVisit(body_j, block=b, ssa_vals=d)
            else:
                ietxdsl_functions.myVisit(body_i, block=b, ssa_vals=d)

        # print Kernel

        # Add frees
        for op_free in op.body.frees:
            ietxdsl_functions.myVisit(op_free, block=b, ssa_vals=d)

        cgen.printCallable(call_obj)
        module.regions[0].blocks[0].add_op(call_obj)

    from xdsl.printer import Printer
    Printer().print(module)
    return cgen.str()


def _op_to_func(op: Operator):
    # Visit the Operator body
    assert isinstance(op.body, CallableBody)


    # Scan an Operator
    # Those parameters without associated types aren't printed in the Kernel header
    # # import pdb;pdb.set_trace()
    op_symbols = FindSymbols('defines').visit(op)
    op_param_names = [s._C_name for s in op_symbols]
    op_header_params = [i._C_name for i in list(op.parameters)]
    op_types = [i._C_typename for i in list(op.parameters)]
    op_type_qs = [i._C_type_qualifier for i in list(op.parameters)]
    prefix = '-'.join(op.prefix)
    retv = str(op.retval)

    # # import pdb;pdb.set_trace()

    # Game is here we start a dict from op params, focus
    arg_types = get_arg_types(op.parameters)
    # b = Block([i32] * len(op_param_names))
    block = Block(arg_types)
    ssa_val_dict = {param._C_name: val for param, val in zip(op.parameters, block.args)}

    # Add Casts
    for cast in FindNodes(PointerCast).visit(op.body):
        ietxdsl_functions.myVisit(cast, block=block, ssa_vals=ssa_val_dict)

    for i in op.body.body:
        # Comments
        # # import pdb;pdb.set_trace()
        if i.args.get('body') != ():
            for body_j in i.body:
                # Casts
                ietxdsl_functions.myVisit(body_j, block=block, ssa_vals=ssa_val_dict)
        else:
            ietxdsl_functions.myVisit(i, block=block, ssa_vals=ssa_val_dict)

    # add a trailing return
    block.add_op(func.Return())

    func_op = func.FuncOp.from_region(str(op.name), arg_types, [], Region([block]))

    func_op.attributes['param_names'] = builtin.ArrayAttr([
        builtin.StringAttr(str(param._C_name)) for param in op.parameters
    ])

    return func_op


def transform_devito_to_iet_ssa(op: Operator):
    # Check for the existence of funcs in the operator (print devito metacalls)
    op_funcs = [value for _, value in op._func_table.items()]
    # Print calls
    call_obj = _op_to_func(op)

    # After finishing kernels, now we check the rest of the functions
    module = builtin.ModuleOp.from_region_or_ops([call_obj])
    for op_func in op_funcs:
        op = op_func.root
        name = op.name

        # Those parameters without associated types aren't printed in the Kernel header
        op_param_names = [s._C_name for s in FindSymbols('defines').visit(op)]
        op_header_params = [i._C_name for i in list(op.parameters)]
        op_types = [i._C_typename for i in list(op.parameters)]
        op_type_qs = [i._C_type_qualifier for i in list(op.parameters)]
        prefix = '-'.join(op.prefix)
        retval = str(op.retval)
        b = Block([i32] * len(op_param_names))
        d = {name: register for name, register in zip(op_param_names, b.args)}

        # Add Allocs
        for op_alloc in op.body.allocs:
            ietxdsl_functions.myVisit(op_alloc, block=b, ssa_vals=d)

        # Add obj defs
        for op_obj in op.body.objs:
            ietxdsl_functions.myVisit(op_obj, block=b, ssa_vals=d)

        # Add Casts
        for cast in FindNodes(PointerCast).visit(op.body):
            ietxdsl_functions.myVisit(cast, block=b, ssa_vals=d)

        call_obj = Callable.get(name, op_param_names, op_header_params, op_types,
                                op_type_qs, retval, prefix, b)

        for body_i in op.body.body:
            # Comments
            if body_i.args.get('body') != ():
                for body_j in body_i.body:
                    # Casts
                    ietxdsl_functions.myVisit(body_j, block=b, ssa_vals=d)
            else:
                ietxdsl_functions.myVisit(body_i, block=b, ssa_vals=d)

        # print Kernel

        # Add frees
        for op_free in op.body.frees:
            ietxdsl_functions.myVisit(op_free, block=b, ssa_vals=d)

        module.regions[0].blocks[0].add_op(call_obj)

    return module
