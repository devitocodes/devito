from devito import Operator

from devito.ir import PointerCast, FindNodes
from devito.ir.iet import FindSymbols
from devito.ir.iet.nodes import CallableBody

from devito.ir.ietxdsl import (MLContext, IET, Block, CGeneration,
                               ietxdsl_functions, Callable)

from devito.ir.ietxdsl.ietxdsl_functions import collectStructs
from xdsl.dialects.builtin import Builtin, i32


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

    call_obj = visit_Operator(op)
    cgen.printCallable(call_obj)

    # Look for extra functions in the operator and print them out
    # TODO print their definition on top of the code
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

        b = Block.from_arg_types([i32] * len(op_param_names))
        d = {name: register for name, register in zip(op_param_names, b.args)}

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

        cgen.printCallable(call_obj)

    from xdsl.printer import Printer
    Printer().print(call_obj.body)

    return cgen.str()


def visit_Operator(op):
    # Scan an Operator
    # Those parameters without associated types aren't printed in the Kernel header
    op_funcs = [value for _, value in op._func_table.items()]
    # import pdb;pdb.set_trace()

    op_symbols = FindSymbols('defines').visit(op)
    op_param_names = [s._C_name for s in op_symbols]
    op_header_params = [i._C_name for i in list(op.parameters)]
    op_types = [i._C_typename for i in list(op.parameters)]
    op_type_qs = [i._C_type_qualifier for i in list(op.parameters)]
    prefix = '-'.join(op.prefix)
    retv = str(op.retval)

    # import pdb;pdb.set_trace()

    # TOFIX
    b = Block.from_arg_types([i32] * len(op_param_names))
    d = {name: register for name, register in zip(op_param_names, b.args)}

    # Add Casts
    for cast in FindNodes(PointerCast).visit(op.body):
        ietxdsl_functions.myVisit(cast, block=b, ctx=d)

    # Create a Callable for the main kernel
    call_obj = Callable.get(str(op.name), op_param_names, op_header_params, op_types,
                            op_type_qs, retv, prefix, b)

    # Visit the Operator body
    assert isinstance(op.body, CallableBody)
    for i in op.body.body:
        # Comments
        # import pdb;pdb.set_trace()
        if i.args.get('body') != ():
            for body_j in i.body:
                # Casts
                ietxdsl_functions.myVisit(body_j, block=b, ctx=d)
        else:
            ietxdsl_functions.myVisit(i, block=b, ctx=d)

    # print Kernel
    return call_obj
