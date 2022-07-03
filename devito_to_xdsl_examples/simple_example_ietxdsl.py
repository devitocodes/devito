# continuation of xdsl generated code pulled out from GenerateXDSL jupyter notebook
from devito import Grid, TimeFunction, Eq, Operator
from devito.ir import retrieve_iteration_tree
from devito.tools import flatten

from devito.ir import PointerCast, FindNodes

from devito.ir.ietxdsl import (MLContext, Builtin, IET, Block, CGeneration, Statement,
                               StructDecl, ietxdsl_functions, Callable)

# Define a simple Devito Operator
grid = Grid(shape=(3, 3))
u = TimeFunction(name='u', grid=grid)
eq = Eq(u.forward, u + 1)
op = Operator([eq])
op.apply(time_M=5)


def createStatement(initial_string, val):
    ret_str = initial_string
    if isinstance(val, tuple):
        for t in val:
            ret_str = ret_str + " " + t
    else:
        ret_str = ret_str + " " + val

    return ret_str


ctx = MLContext()
Builtin(ctx)
iet = IET(ctx)

# Those parameters without associated types aren't printed in the Kernel header
op_params = list(op.parameters)
op_param_names = [opi._C_name for opi in op_params]
op_header_params = [opi._C_name for opi in op_params]
op_types = [opi._C_typename for opi in op_params]

# we still need to add the extra time indices even though they aren't passed in

devito_iterations = flatten(retrieve_iteration_tree(op.body))
timing_indices = [i.uindices for i in devito_iterations if i.dim.is_Time]
op_param_names.append(str(t) for t in timing_indices)

b = Block.from_arg_types([iet.i32] * len(op_param_names))
d = {name: register for name, register in zip(op_param_names, b.args)}

# TODO: should this all be pulled out from just "op"?
headers = op._headers
includes = op._includes
struct_decs = [
    i._C_typedecl for i in op.parameters if i._C_typedecl is not None
]

kernel_comments = op.body.body[0]
uvec_cast = FindNodes(PointerCast).visit(op)[0]
full_loop = op.body.body[1].args.get('body')[0]

comment_result = ietxdsl_functions.myVisit(kernel_comments, block=b, ctx=d)
uvec_result = ietxdsl_functions.myVisit(uvec_cast, block=b, ctx=d)
main_result = ietxdsl_functions.myVisit(full_loop, block=b, ctx=d)

call_obj = Callable.get("kernel", op_param_names, op_header_params, op_types,
                        b)
cgen = CGeneration()

# TODO: is there a more formal way to do this without cherry-picking from op?
# print headers:
for header in headers:
    cgen.printOperation(Statement.get(createStatement("#define", header)))
# print includes:
for include in includes:
    # TOFIX double quotes
    cgen.printOperation(Statement.get(createStatement("#include ", include)))
# print structs:
for struct in struct_decs:
    cgen.printOperation(
        StructDecl.get(struct.tpname, struct.fields, struct.declname,
                       struct.pad_bytes))

# print Kernel
cgen.printCallable(call_obj)

print(cgen.str())

# Printer()._print_named_block(b)
