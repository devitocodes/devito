from devito import Operator

from devito.ir import PointerCast, FindNodes

from devito.ir.ietxdsl import (MLContext, Builtin, IET, Block, CGeneration,
                               ietxdsl_functions, Callable)

from devito.ir.ietxdsl.ietxdsl_functions import collectStructs, getOpParamsNames


def transform_devito_xdsl_string(op: Operator):

    ctx = MLContext()
    Builtin(ctx)
    iet = IET(ctx)

    # Those parameters without associated types aren't printed in the Kernel header
    op_params = list(op.parameters)
    op_param_names = [opi._C_name for opi in op_params]
    op_header_params = [opi._C_name for opi in op_params]
    op_types = [opi._C_typename for opi in op_params]

    # need to determine how large 'body' is
    body = op.body.body[1].args.get('body')
    body_size = len(body)

    # amass all required parameters
    op_param_names = getOpParamsNames(op, op_param_names)

    b = Block.from_arg_types([iet.i32] * len(op_param_names))
    d = {name: register for name, register in zip(op_param_names, b.args)}

    kernel_comments = op.body.body[0]
    ietxdsl_functions.myVisit(kernel_comments, block=b, ctx=d)

    num_pointer_casts = len(FindNodes(PointerCast).visit(op))

    for i in range(0, num_pointer_casts):
        uvec_cast = FindNodes(PointerCast).visit(op)[i]
        ietxdsl_functions.myVisit(uvec_cast, block=b, ctx=d)

    for i in range(0, body_size):
        section = body[i]
        ietxdsl_functions.myVisit(section, block=b, ctx=d)

    call_obj = Callable.get("kernel", op_param_names, op_header_params, op_types,
                            b)
    cgen = CGeneration()

    # print header information
    ietxdsl_functions.printHeaders(cgen, "#define", op._headers)
    ietxdsl_functions.printHeaders(cgen, "#include ", op._includes)  # TOFIX double quotes
    ietxdsl_functions.printStructs(cgen, collectStructs(op.parameters))

    # print Kernel
    cgen.printCallable(call_obj)

    return cgen.str()
