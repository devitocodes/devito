from devito import Grid, TimeFunction, Eq, Operator
from devito.ir.ietxdsl.xdsl_passes import transform_devito_xdsl_string

# flake8: noqa

def test_udx_conversion():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    op = Operator([eq])
    op.apply(time_M=5)

    xdsl_string = transform_devito_xdsl_string(op)

    assert xdsl_string == ("#define _POSIX_C_SOURCE 200809L\n"
                          "#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);\n"
                          "#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;\n"
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
                                   "struct profiler\n"
                                    "{\n"
                                    "double section0;\n"
                                    "};\n"
                                   "int Kernel(struct dataobj * u_vec,const float h_x,const int time_M,"
                                                                      "const int "
                                   "time_m,const int x_M,const int x_m,const int y_M,const int y_m,struct profiler * timers){\n"
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
                                   "    START_TIMER(section0)\n"
                                   "for (int x = x_m; x <= x_M; x += 1)\n"
                                   "{\n"
                                   "  #pragma omp simd aligned(u:32)\n"
                                   "  for (int y = y_m; y <= y_M; y += 1)\n"
                                   "  {\n"
                                   "    u[t1][x + 1][y + 1] = r0*(-u[t0][x + 1][y + 1]) + r0*u[t0][x + 2][y + 1];\n"
                                   "  }\n"
                                   "}\n"
                                   "STOP_TIMER(section0,timers)\n"
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

    xdsl_string = transform_devito_xdsl_string(op)

    assert xdsl_string == ("#define _POSIX_C_SOURCE 200809L\n"
                          "#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);\n"
                          "#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;\n"
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
                                   "struct profiler\n"
                                   "{\n"
                                   "double section0;\n"
                                   "};\n"
                                   "int Kernel(struct dataobj * u_vec,const int time_M,"
                                   "const int "
                                   "time_m,const int x_M,const int x_m,const int y_M,const int y_m,struct profiler * timers){\n"
                                   "  /* Flush denormal numbers to zero in hardware */\n"
                                   "  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);\n"
                                   "  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);\n"
                                   "  \n"
                                   "  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ "
                                   "((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;\n"
                                   "  for (int time = time_m,   t0 = (time)%(2)  ,  t1 = (time + 1)%(2)  ; "
                                   "time <= time_M;   time += 1,   t0 = (time)%(2)  ,  t1 = (time + 1)%(2)  )  {\n"
                                   "    /* Begin section0 */\n"
                                   "    START_TIMER(section0)\n"
                                   "for (int x = x_m; x <= x_M; x += 1)\n"
                                   "{\n"
                                   "  #pragma omp simd aligned(u:32)\n"
                                   "  for (int y = y_m; y <= y_M; y += 1)\n"
                                   "  {\n"
                                   "    u[t1][x + 1][y + 1] = u[t0][x + 1][y + 1] + 1;\n"
                                   "  }\n"
                                   "}\n"
                                   "STOP_TIMER(section0,timers)\n"
                                   "    /* End section0 */\n"
                                   "  }\n"
                                   "  return 0;\n"
                                   "}\n")

def test_u_and_v_conversion():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid, time_order=2)
    v = TimeFunction(name='v', grid=grid, time_order=2)
    eq0 = Eq(u.forward, u.dt2)
    eq1 = Eq(v.forward, u.dt2)
    op = Operator([eq0, eq1])

    xdsl_string = transform_devito_xdsl_string(op)

    assert xdsl_string == ("#define _POSIX_C_SOURCE 200809L\n"
                          "#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);\n"
                          "#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;\n"
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
                          "struct profiler\n"
                          "{\n"
                          "double section0;\n"
                          "};\n"
                          "int Kernel(struct dataobj * u_vec,struct dataobj * v_vec,const float dt,const int time_M,"
                          "const int "
                          "time_m,const int x_M,const int x_m,const int y_M,const int y_m,struct profiler * timers){\n"
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
                          "    START_TIMER(section0)\n"
                          "for (int x = x_m; x <= x_M; x += 1)\n"
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
                          "STOP_TIMER(section0,timers)\n"
                          "    /* End section0 */\n"
                          "  }\n"
                          "  return 0;\n"
                          "}\n")
