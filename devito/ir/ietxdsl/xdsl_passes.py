from devito import Operator

from devito.ir import PointerCast, FindNodes
from devito.ir.iet import FindSymbols

from devito.ir.ietxdsl import (MLContext, IET, Block, CGeneration,
                               ietxdsl_functions, Callable)

from devito.ir.ietxdsl.ietxdsl_functions import collectStructs
from xdsl.dialects.builtin import Builtin


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

    # Those parameters without associated types aren't printed in the Kernel header
    op_param_names = [s._C_name for s in FindSymbols('defines').visit(op)]
    op_header_params = [opi._C_name for opi in list(op.parameters)]
    op_types = [opi._C_typename for opi in list(op.parameters)]

    b = Block.from_arg_types([iet.i32] * len(op_param_names))
    d = {name: register for name, register in zip(op_param_names, b.args)}

    call_obj = Callable.get("kernel", op_param_names, op_header_params, op_types, b)
    cgen = CGeneration()
    # print header information
    ietxdsl_functions.printHeaders(cgen, "#define", op._headers)
    ietxdsl_functions.printIncludes(cgen, "#include", op._includes)
    ietxdsl_functions.printStructs(cgen, collectStructs(op.parameters))

    for body_i in op.body.body:
        # Comments
        if body_i.args.get('body') == ():
            ietxdsl_functions.myVisit(body_i, block=b, ctx=d)
        else:
            # Casts
            for cast in FindNodes(PointerCast).visit(op.body):
                ietxdsl_functions.myVisit(cast, block=b, ctx=d)

        for body_j in body_i.body:
            ietxdsl_functions.myVisit(body_j, block=b, ctx=d)

    # print Kernel
    cgen.printCallable(call_obj)

    return cgen.str()
