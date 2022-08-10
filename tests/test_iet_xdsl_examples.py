from devito import Grid, TimeFunction, Eq, Operator
from devito.ir import retrieve_iteration_tree
from devito.tools import flatten

from devito.ir import PointerCast, FindNodes

from devito.ir.ietxdsl import (MLContext, Builtin, IET, Block, CGeneration,
                               Statement, StructDecl, ietxdsl_functions,
                               Callable)

from devito.ir.ietxdsl.ietxdsl_functions import createStatement, collectStructs


# flake8: noqa

# Currently fails due to lingering float issue
def test_udx_conversion():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    op = Operator([eq])
    op.apply(time_M=5)

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

    # then add in anything that comes before the main loop:
    for i in range(0, (body_size - 1)):
        val = op.body.body[1].args.get('body')[0]._args['body'][0].write
        op_param_names.append(str(val))

    # we also need to pass in the extra variables
    num_extra_expressions = len(op.body.body[1].args.get('body'))

    # we still need to add the extra time indices even though they aren't passed in
    devito_iterations = flatten(retrieve_iteration_tree(op.body))
    timing_indices = [i.uindices for i in devito_iterations if i.dim.is_Time]
    for tup in timing_indices:
        for t in tup:
            op_param_names.append((str(t)))

    b = Block.from_arg_types([iet.i32] * len(op_param_names))
    d = {name: register for name, register in zip(op_param_names, b.args)}

    headers = op._headers
    includes = op._includes

    struct_decs = collectStructs(op.parameters)

    kernel_comments = op.body.body[0]
    uvec_cast = FindNodes(PointerCast).visit(op)[0]

    comment_result = ietxdsl_functions.myVisit(kernel_comments, block=b, ctx=d)
    uvec_result = ietxdsl_functions.myVisit(uvec_cast, block=b, ctx=d)

    for i in range(0, body_size):
        section = op.body.body[1].args.get('body')[i]
        section_result = ietxdsl_functions.myVisit(section, block=b, ctx=d)

    call_obj = Callable.get("kernel", op_param_names, op_header_params, op_types,
                            b)
    cgen = CGeneration()

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

    assert cgen.str() == ("#define _POSIX_C_SOURCE 200809L\n"
                                   "#include  stdlib.h\n"
                                   "#include  math.h\n"
                                   "#include  sys/time.h\n"
                                   "#include  xmmintrin.h\n"
                                   "#include  pmmintrin.h\n"
                                   "struct dataobj\n"
                                   "{\n"
                                   "void *restrict data;\n"
                                   "unsigned long * size;\n"
                                   "unsigned long * npsize;\n"
                                   "unsigned long * dsize;\n"
                                   "int * hsize;\n"
                                   "int * hofs;\n"
                                   "int * oofs;\n"
                                   "};\n"
                                   "int Kernel(struct dataobj * u_vec,const float h_x,const int time_M,"
                                                                      "const int "
                                   "time_m,const int x_M,const int x_m,const int y_M,const int y_m){\n"
                                   "  /* Flush denormal numbers to zero in hardware */\n"
                                   "  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);\n"
                                   "  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);\n"
                                   "  \n"
                                   "  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ "
                                   "((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;\n"
                                   "  float r0 = (h_x) ^ (-1);\n"
                                   "  for (int time = time_m,   t0 = (time)%(2)  ,  t1 = (time + 1)%(2)  ; "
                                   "time <= time_M;   time += 1,   t0 = (time)%(2)  ,  t1 = (time + 1)%(2)  )  {\n"
                                   "    /* Begin section0 */\n"
                                   "    for (int x = x_m; x <= x_M; x += 1)\n"
                                   "{\n"
                                   "  #pragma omp simd aligned(u:32)\n"
                                   "  for (int y = y_m; y <= y_M; y += 1)\n"
                                   "  {\n"
                                   "    u[t1][x + 1][y + 1] = r0*(-u[t0][x + 1][y + 1]) + r0*u[t0][x + 2][y + 1];\n"
                                   "  }\n"
                                   "}\n"
                                   "    /* End section0 */\n"
                                   "  }\n"
                                   "  return 0;\n"
                                     "}\n")

def test_u_plus1_conversion():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u + 1)
    op = Operator([eq])
    op.apply(time_M=5)

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
    for tup in timing_indices:
        for t in tup:
            op_param_names.append((str(t)))

    b = Block.from_arg_types([iet.i32] * len(op_param_names))
    d = {name: register for name, register in zip(op_param_names, b.args)}

    headers = op._headers
    includes = op._includes
    struct_decs = collectStructs(op.parameters)

    kernel_comments = op.body.body[0]
    uvec_cast = FindNodes(PointerCast).visit(op)[0]
    full_loop = op.body.body[1].args.get('body')[0]

    comment_result = ietxdsl_functions.myVisit(kernel_comments, block=b, ctx=d)
    uvec_result = ietxdsl_functions.myVisit(uvec_cast, block=b, ctx=d)
    main_result = ietxdsl_functions.myVisit(full_loop, block=b, ctx=d)

    call_obj = Callable.get("kernel", op_param_names, op_header_params, op_types,
                            b)
    cgen = CGeneration()

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

    assert cgen.str() == ("#define _POSIX_C_SOURCE 200809L\n"
                                   "#include  stdlib.h\n"
                                   "#include  math.h\n"
                                   "#include  sys/time.h\n"
                                   "#include  xmmintrin.h\n"
                                   "#include  pmmintrin.h\n"
                                   "struct dataobj\n"
                                   "{\n"
                                   "void *restrict data;\n"
                                   "unsigned long * size;\n"
                                   "unsigned long * npsize;\n"
                                   "unsigned long * dsize;\n"
                                   "int * hsize;\n"
                                   "int * hofs;\n"
                                   "int * oofs;\n"
                                   "};\n"
                                   "int Kernel(struct dataobj * u_vec,const int time_M,"
                                   "const int "
                                   "time_m,const int x_M,const int x_m,const int y_M,const int y_m){\n"
                                   "  /* Flush denormal numbers to zero in hardware */\n"
                                   "  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);\n"
                                   "  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);\n"
                                   "  \n"
                                   "  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ "
                                   "((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;\n"
                                   "  for (int time = time_m,   t0 = (time)%(2)  ,  t1 = (time + 1)%(2)  ; "
                                   "time <= time_M;   time += 1,   t0 = (time)%(2)  ,  t1 = (time + 1)%(2)  )  {\n"
                                   "    /* Begin section0 */\n"
                                   "    for (int x = x_m; x <= x_M; x += 1)\n"
                                   "{\n"
                                   "  #pragma omp simd aligned(u:32)\n"
                                   "  for (int y = y_m; y <= y_M; y += 1)\n"
                                   "  {\n"
                                   "    u[t1][x + 1][y + 1] = u[t0][x + 1][y + 1] + 1;\n"
                                   "  }\n"
                                   "}\n"
                                   "    /* End section0 */\n"
                                   "  }\n"
                                   "  return 0;\n"
                                   "}\n")

# Currently fails due to lingering float issue
def test_u_and_v_conversion():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid, time_order=2)
    v = TimeFunction(name='v', grid=grid, time_order=2)
    eq0 = Eq(u.forward, u.dt2)
    eq1 = Eq(v.forward, u.dt2)
    op = Operator([eq0, eq1])

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

    # then add in anything that comes before the main loop:
    for i in range(0, (body_size - 1)):
        val = op.body.body[1].args.get('body')[0]._args['body'][0].write
        op_param_names.append(str(val))

    # we also need to pass in the extra variables
    num_extra_expressions = len(op.body.body[1].args.get('body'))

    # we still need to add the extra time indices even though they aren't passed in
    devito_iterations = flatten(retrieve_iteration_tree(op.body))
    timing_indices = [i.uindices for i in devito_iterations if i.dim.is_Time]
    for tup in timing_indices:
        for t in tup:
            op_param_names.append((str(t)))

    b = Block.from_arg_types([iet.i32] * len(op_param_names))
    d = {name: register for name, register in zip(op_param_names, b.args)}

    headers = op._headers
    includes = op._includes
    struct_decs = collectStructs(op.parameters)

    kernel_comments = op.body.body[0]
    comment_result = ietxdsl_functions.myVisit(kernel_comments, block=b, ctx=d)

    num_pointer_casts = len(FindNodes(PointerCast).visit(op))

    for i in range(0, num_pointer_casts):
        uvec_cast = FindNodes(PointerCast).visit(op)[i]
        uvec_result = ietxdsl_functions.myVisit(uvec_cast, block=b, ctx=d)

    for i in range(0, body_size):
        section = op.body.body[1].args.get('body')[i]
        section_result = ietxdsl_functions.myVisit(section, block=b, ctx=d)

    call_obj = Callable.get("kernel", op_param_names, op_header_params, op_types,
                            b)
    cgen = CGeneration()

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

    assert cgen.str() == ("#define _POSIX_C_SOURCE 200809L\n"
                          "#include  stdlib.h\n"
                          "#include  math.h\n"
                          "#include  sys/time.h\n"
                          "#include  xmmintrin.h\n"
                          "#include  pmmintrin.h\n"
                          "struct dataobj\n"
                          "{\n"
                          "void *restrict data;\n"
                          "unsigned long * size;\n"
                          "unsigned long * npsize;\n"
                          "unsigned long * dsize;\n"
                          "int * hsize;\n"
                          "int * hofs;\n"
                          "int * oofs;\n"
                          "};\n"
                          "int Kernel(struct dataobj * u_vec,struct dataobj * v_vec,const float dt,const int time_M,"
                          "const int "
                          "time_m,const int x_M,const int x_m,const int y_M,const int y_m){\n"
                          "  /* Flush denormal numbers to zero in hardware */\n"
                          "  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);\n"
                          "  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);\n"
                          "  \n"
                          "  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ "
                          "((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;\n"
                          "  float (*restrict v)[v_vec->size[1]][v_vec->size[2]] __attribute__ "
                          "((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]]) v_vec->data;\n"
                          "  float r0 = (dt * dt) ^ (-1);\n"
                          "  for (int time = time_m,   t0 = (time)%(3)  ,  t1 = (time + 2)%(3)  ,  t2 = "
                          "(time + 1)%(3)  ;"
                          " time <= time_M;   time += 1,   t0 = (time)%(3)  ,  t1 = (time + 2)%(3)  ,  t2 = (time + 1)%(3)  )  {\n"
                          "    /* Begin section0 */\n"
                          "    for (int x = x_m; x <= x_M; x += 1)\n"
                          "{\n"
                          "  #pragma omp simd aligned(u,v:32)\n"
                          "  for (int y = y_m; y <= y_M; y += 1)\n"
                          "  {\n"
                          "    float r2 = r0*u[t1][x + 1][y + 1];\n"
                          "    float r1 = r0*(-2.0F*u[t0][x + 1][y + 1]);\n"
                          "    u[t2][x + 1][y + 1] = r0*u[t2][x + 1][y + 1] + r1 + r2;\n"
                          "    v[t2][x + 1][y + 1] = r0*u[t2][x + 1][y + 1] + r1 + r2;\n"
                          "  }\n"
                          "}\n"
                          "    /* End section0 */\n"
                          "  }\n"
                          "  return 0;\n"
                          "}\n")
