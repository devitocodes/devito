{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Domain, Halo and Padding regions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we will learn how the `Halo` and `Padding` regions affect the Operator construction. For that, we will use the time marching example shown in [01_iet](https://github.com/opesci/devito/blob/master/examples/compiler/01_iet.ipynb) tutorial. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from devito import Eq, Grid, TimeFunction, Operator\n",
    "\n",
    "grid = Grid(shape=(3, 3))\n",
    "u = TimeFunction(name='u', grid=grid)\n",
    "u.data[:] = 1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "At this moment, we have a time-varying 3x3 grid filled with `1's`. Below, we can see the `domain` data values:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1. 1. 1.]\n",
      "  [1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]\n",
      "  [1. 1. 1.]]]\n"
     ]
    }
   ],
   "source": [
    "print(u.data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We now create an `Operator` that increments by `2` all points in the computational domain at each timestep."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from devito import configuration\n",
    "configuration['openmp'] = 0\n",
    "\n",
    "eq = Eq(u.forward, u+2)\n",
    "op = Operator(eq)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finally, we can print the `op` to see the generated code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "#define _POSIX_C_SOURCE 200809L\n",
      "#include \"stdlib.h\"\n",
      "#include \"math.h\"\n",
      "#include \"sys/time.h\"\n",
      "#include \"xmmintrin.h\"\n",
      "#include \"pmmintrin.h\"\n",
      "\n",
      "struct profiler\n",
      "{\n",
      "  double section0;\n",
      "} ;\n",
      "\n",
      "\n",
      "int Kernel(float *restrict u_vec, const int time_M, const int time_m, struct profiler* timers, const int x_M, const int x_m, const int x_size, const int y_M, const int y_m, const int y_size)\n",
      "{\n",
      "  float (*restrict u)[x_size + 1 + 1][y_size + 1 + 1] __attribute__((aligned(64))) = (float (*)[x_size + 1 + 1][y_size + 1 + 1]) u_vec;\n",
      "  /* Flush denormal numbers to zero in hardware */\n",
      "  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);\n",
      "  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);\n",
      "  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))\n",
      "  {\n",
      "    struct timeval start_section0, end_section0;\n",
      "    gettimeofday(&start_section0, NULL);\n",
      "    for (int x = x_m; x <= x_M; x += 1)\n",
      "    {\n",
      "      #pragma omp simd\n",
      "      for (int y = y_m; y <= y_M; y += 1)\n",
      "      {\n",
      "        u[t1][x + 1][y + 1] = u[t0][x + 1][y + 1] + 2;\n",
      "      }\n",
      "    }\n",
      "    gettimeofday(&end_section0, NULL);\n",
      "    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;\n",
      "  }\n",
      "  return 0;\n",
      "}\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(op)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "When we take a look at the constructed expression ($u[t1][x + 1][y + 1] = u[t0][x + 1][y + 1] + 2$) we see `+1` being added to the spatial indexes of `u`."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "That occurs because there are grid points surrounding the `domain region`, i.e. ghost points that are accessed by the stencil when iterating in the proximity of the domain boundary. Those points represent a region called `Halo`. The Halo region can be seen below, it's the zeros surrounding the domain region."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[0. 0. 0. 0. 0.]\n",
      "  [0. 1. 1. 1. 0.]\n",
      "  [0. 1. 1. 1. 0.]\n",
      "  [0. 1. 1. 1. 0.]\n",
      "  [0. 0. 0. 0. 0.]]\n",
      "\n",
      " [[0. 0. 0. 0. 0.]\n",
      "  [0. 1. 1. 1. 0.]\n",
      "  [0. 1. 1. 1. 0.]\n",
      "  [0. 1. 1. 1. 0.]\n",
      "  [0. 0. 0. 0. 0.]]]\n"
     ]
    }
   ],
   "source": [
    "print(u.data_with_halo)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Thus, the array accesses become logically aligned to the equation’s natural domain. For instance, given the usual Function $u(t, x, y)$ having one point on each side of the `x` and `y` halo regions, the array accesses $u[t, x, y]$ and $u[t, x + 2, y + 2]$ are transformed, respectively, into $u[t, x + 1, y + 1]$ and $u[t, x + 3, y + 3]$. When $x = y = 0$, therefore, the values $u[t, 1, 1]$ and $u[t, 3, 3]$ are fetched, representing the first and third points in the computational domain. \n",
    "\n",
    "By default, the Halo region has `1` point on each side of the space dimensions. Sometimes, those points may be unnecessary. On the other hand, for instance, depending on the PDE being approximated, more points may be necessary. Thus, this default value can be changed by passing a value in `space_order`:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1. 1. 1.]\n",
      "  [1. 1. 1.]\n",
      "  [1. 1. 1.]]\n",
      "\n",
      " [[1. 1. 1.]\n",
      "  [1. 1. 1.]\n",
      "  [1. 1. 1.]]]\n"
     ]
    }
   ],
   "source": [
    "u0 = TimeFunction(name='u0', grid=grid, space_order=0)\n",
    "u0.data[:] = 1\n",
    "print(u0.data_with_halo)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]]\n",
      "\n",
      " [[0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]]]\n"
     ]
    }
   ],
   "source": [
    "u2 = TimeFunction(name='u2', grid=grid, space_order=2)\n",
    "u2.data[:] = 1\n",
    "print(u2.data_with_halo)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Also, one can pass a 3-tuple `(o, lp, rp)` instead of a single integer representing the discretization order. Here, `o` is the discretization order, while `lp` and `rp` indicate how many points are expected on left (lp) and right (rp) of a point of interest."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "u_new = TimeFunction(name='u_new', grid=grid, space_order=(4, 3, 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 1. 1. 1. 0.]\n",
      "  [0. 0. 0. 1. 1. 1. 0.]\n",
      "  [0. 0. 0. 1. 1. 1. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]]\n",
      "\n",
      " [[0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 1. 1. 1. 0.]\n",
      "  [0. 0. 0. 1. 1. 1. 0.]\n",
      "  [0. 0. 0. 1. 1. 1. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]]]\n"
     ]
    }
   ],
   "source": [
    "u_new.data[:] = 1\n",
    "print(u_new.data_with_halo)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's write the code again, but, this time, changing the `space_order` values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "#define _POSIX_C_SOURCE 200809L\n",
      "#include \"stdlib.h\"\n",
      "#include \"math.h\"\n",
      "#include \"sys/time.h\"\n",
      "#include \"xmmintrin.h\"\n",
      "#include \"pmmintrin.h\"\n",
      "\n",
      "struct profiler\n",
      "{\n",
      "  double section0;\n",
      "} ;\n",
      "\n",
      "\n",
      "int Kernel(float *restrict u_new_vec, const int time_M, const int time_m, struct profiler* timers, const int x_M, const int x_m, const int x_size, const int y_M, const int y_m, const int y_size)\n",
      "{\n",
      "  float (*restrict u_new)[x_size + 1 + 3][y_size + 1 + 3] __attribute__((aligned(64))) = (float (*)[x_size + 1 + 3][y_size + 1 + 3]) u_new_vec;\n",
      "  /* Flush denormal numbers to zero in hardware */\n",
      "  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);\n",
      "  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);\n",
      "  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))\n",
      "  {\n",
      "    struct timeval start_section0, end_section0;\n",
      "    gettimeofday(&start_section0, NULL);\n",
      "    for (int x = x_m; x <= x_M; x += 1)\n",
      "    {\n",
      "      #pragma omp simd\n",
      "      for (int y = y_m; y <= y_M; y += 1)\n",
      "      {\n",
      "        u_new[t1][x + 3][y + 3] = u_new[t0][x + 3][y + 3] + 2;\n",
      "      }\n",
      "    }\n",
      "    gettimeofday(&end_section0, NULL);\n",
      "    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;\n",
      "  }\n",
      "  return 0;\n",
      "}\n",
      "\n"
     ]
    }
   ],
   "source": [
    "equation = Eq(u_new.forward, u_new + 2)\n",
    "op = Operator(equation)\n",
    "print(op)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finally, let's run the operator."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Operator `Kernel` run in 0.00 s\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 5. 5. 5. 0.]\n",
      "  [0. 0. 0. 5. 5. 5. 0.]\n",
      "  [0. 0. 0. 5. 5. 5. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]]\n",
      "\n",
      " [[0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 7. 7. 7. 0.]\n",
      "  [0. 0. 0. 7. 7. 7. 0.]\n",
      "  [0. 0. 0. 7. 7. 7. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0.]]]\n"
     ]
    }
   ],
   "source": [
    "op.apply(time_M=2)\n",
    "print(u_new.data_with_halo)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The halo is surrounded by another data region, called `padding`. Padding can be used for data alignment. By default, there is no padding, but it can be changed by passing a value in `padding`, as shown below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "#define _POSIX_C_SOURCE 200809L\n",
      "#include \"stdlib.h\"\n",
      "#include \"math.h\"\n",
      "#include \"sys/time.h\"\n",
      "#include \"xmmintrin.h\"\n",
      "#include \"pmmintrin.h\"\n",
      "\n",
      "struct profiler\n",
      "{\n",
      "  double section0;\n",
      "} ;\n",
      "\n",
      "\n",
      "int Kernel(float *restrict u_pad_vec, const int time_M, const int time_m, struct profiler* timers, const int x_M, const int x_m, const int x_size, const int y_M, const int y_m, const int y_size)\n",
      "{\n",
      "  float (*restrict u_pad)[x_size + 2 + 2 + 2 + 2][y_size + 2 + 2 + 2 + 2] __attribute__((aligned(64))) = (float (*)[x_size + 2 + 2 + 2 + 2][y_size + 2 + 2 + 2 + 2]) u_pad_vec;\n",
      "  /* Flush denormal numbers to zero in hardware */\n",
      "  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);\n",
      "  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);\n",
      "  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))\n",
      "  {\n",
      "    struct timeval start_section0, end_section0;\n",
      "    gettimeofday(&start_section0, NULL);\n",
      "    for (int x = x_m; x <= x_M; x += 1)\n",
      "    {\n",
      "      #pragma omp simd\n",
      "      for (int y = y_m; y <= y_M; y += 1)\n",
      "      {\n",
      "        u_pad[t1][x + 4][y + 4] = u_pad[t0][x + 4][y + 4] + 2;\n",
      "      }\n",
      "    }\n",
      "    gettimeofday(&end_section0, NULL);\n",
      "    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;\n",
      "  }\n",
      "  return 0;\n",
      "}\n",
      "\n"
     ]
    }
   ],
   "source": [
    "u_pad = TimeFunction(name='u_pad', grid=grid, space_order=2, padding=(0,2,2))\n",
    "u_pad.data_with_halo[:] = 1\n",
    "u_pad.data[:] = 2\n",
    "equation = Eq(u_pad.forward, u_pad + 2)\n",
    "op = Operator(equation)\n",
    "print(op)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we see `+4` being added on each side of the space dimensions. Of which, +2 belong to space_order (halo) and +2 belong to padding.\n",
    "We can see the data using `data_allocated`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 2. 2. 2. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 2. 2. 2. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 2. 2. 2. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]\n",
      "\n",
      " [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 2. 2. 2. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 2. 2. 2. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 2. 2. 2. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 1. 1. 1. 1. 1. 1. 1. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
      "  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]]\n"
     ]
    }
   ],
   "source": [
    "print(u_pad.data_allocated)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The `data_allocated` property shows the `domain + halo + padding` data values. Above, the domain is filled with 2's, the halo is filled with 1's and the padding is filled with 0's."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
