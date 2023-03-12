from devito import Operator

from devito.ir import PointerCast, FindNodes
from devito.ir.iet import FindSymbols
from devito.ir.iet.nodes import CallableBody, MetaCall, Definition, Dereference, Prodder  # noqa

from devito.ir.ietxdsl import (MLContext, IET, Block, CGeneration,
                               ietxdsl_functions, Callable)

from devito.ir.ietxdsl.ietxdsl_functions import collectStructs, get_arg_types
from xdsl.dialects.builtin import Builtin, i32
from xdsl.dialects.func import FuncOp


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
    call_obj = visit_Operator(op)
    cgen.printCallable(call_obj)

    # Look for extra functions in the operator and print them out
    op_funcs = [value for _, value in op._func_table.items()]

    # After finishing kernels, now we check the rest of the functions
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
        b = Block.from_arg_types([i32] * len(op_param_names))
        d = {name: register for name, register in zip(op_param_names, b.args)}

        # Add Allocs
        for op_alloc in op.body.allocs:
            ietxdsl_functions.myVisit(op_alloc, block=b, ctx=d)

        cgen.print('')
        # Add obj defs
        for op_obj in op.body.objs:
            ietxdsl_functions.myVisit(op_obj, block=b, ctx=d)

        # import pdb;pdb.set_trace()

        # Add Casts
        for cast in FindNodes(PointerCast).visit(op.body):
            ietxdsl_functions.myVisit(cast, block=b, ctx=d)

        call_obj = Callable.get(name, op_param_names, op_header_params, op_types,
                                op_type_qs, retval, prefix, b)

        for body_i in op.body.body:
            # Comments
            if body_i.args.get('body') != ():
                for body_j in body_i.body:
                    # Casts
                    ietxdsl_functions.myVisit(body_j, block=b, ctx=d)
            else:
                ietxdsl_functions.myVisit(body_i, block=b, ctx=d)

        # print Kernel

        # Add frees
        for op_free in op.body.frees:
            ietxdsl_functions.myVisit(op_free, block=b, ctx=d)

        cgen.printCallable(call_obj)

    from xdsl.printer import Printer
    Printer().print(call_obj.body)
    return cgen.str()


def visit_Operator(op):
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
    processed = get_arg_types(op_symbols)
    # b = Block.from_arg_types([i32] * len(op_param_names))
    b = Block.from_arg_types(processed)
    ddict = {name: register for name, register in zip(op_param_names, b.args)}

    # # import pdb;pdb.set_trace()


    # Add Casts
    for cast in FindNodes(PointerCast).visit(op.body):
        ietxdsl_functions.myVisit(cast, block=b, ctx=ddict)

    # Create a Callable for the main kernel
    call_obj = Callable.get(str(op.name), op_param_names, op_header_params, op_types,
                            op_type_qs, retv, prefix, b)

    # Visit the Operator body
    assert isinstance(op.body, CallableBody)
    for i in op.body.body:
        # Comments
        # # import pdb;pdb.set_trace()
        if i.args.get('body') != ():
            for body_j in i.body:
                # Casts
                ietxdsl_functions.myVisit(body_j, block=b, ctx=ddict)
        else:
            ietxdsl_functions.myVisit(i, block=b, ctx=ddict)

    # print Kernel
    from xdsl.printer import Printer
    Printer().print(call_obj.body)
    # import pdb;pdb.set_trace()
    return call_obj
