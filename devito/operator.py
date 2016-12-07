from functools import reduce

import numpy as np
from sympy import Eq, solve

from devito.compiler import get_compiler_from_env
from devito.dimension import t, x, y, z
from devito.dse.inspection import (indexify, retrieve_dimensions,
                                   retrieve_symbols, tolambda)
from devito.dse.symbolics import rewrite
from devito.interfaces import TimeData
from devito.propagator import Propagator

__all__ = ['Operator']


class Operator(object):
    """Class encapsulating a defined operator as defined by the given stencil

    The Operator class is the core abstraction in DeVito that allows
    users to generate high-performance Finite Difference kernels from
    a stencil definition defined from SymPy equations.

    :param nt: Number of timesteps to execute
    :param shape: Shape of the data buffer over which to execute
    :param dtype: Data type for the grid buffer
    :param stencils: SymPy equation or list of equations that define the
                     stencil used to create the kernel of this Operator.
    :param subs: Dict or list of dicts containing the SymPy symbol
                 substitutions for each stencil respectively.
    :param spc_border: Number of spatial padding layers
    :param time_order: Order of the time discretisation
    :param forward: Flag indicating whether to execute forward in time
    :param compiler: Compiler class used to perform JIT compilation.
                     If not provided, the compiler will be inferred from the
                     environment variable DEVITO_ARCH, or default to GNUCompiler.
    :param profile: Flag to enable performance profiling
    :param dse: Set of transformations applied by the Devito Symbolic Engine.
                Available: [None, 'basic', 'advanced' (default)]
    :param cache_blocking: Block sizes used for cache clocking. Can be either a single
                           number used for all dimensions except inner most or a list
                           explicitly stating block sizes for each dimension
                           Set cache_blocking to None to skip blocking on that dim
                           Set cache_blocking to AutoTuner instance, to use auto tuned
                           tuned block sizes
    :param input_params: List of symbols that are expected as input.
    :param output_params: List of symbols that define operator output.
    """
    def __init__(self, nt, shape, dtype=np.float32, stencils=[],
                 subs=[], spc_border=0, time_order=0,
                 forward=True, compiler=None, profile=False, dse='advanced',
                 cache_blocking=None, input_params=None,
                 output_params=None):
        # Derive JIT compilation infrastructure
        self.compiler = compiler or get_compiler_from_env()

        # Ensure stencil and substititutions are lists internally
        self.stencils = stencils if isinstance(stencils, list) else [stencils]
        subs = subs if isinstance(subs, list) else [subs]
        self.input_params = input_params
        self.output_params = output_params

        # Get functions and symbols in LHS/RHS and update params
        sym_undef = set()

        for eqn in self.stencils:
            lhs_def, lhs_undef = retrieve_symbols(eqn.lhs)
            sym_undef.update(lhs_undef)

            if self.output_params is None:
                self.output_params = list(lhs_def)

            rhs_def, rhs_undef = retrieve_symbols(eqn.rhs)
            sym_undef.update(rhs_undef)

            if self.input_params is None:
                self.input_params = list(rhs_def)

        # Pull all dimension indices from the incoming stencil
        dimensions = []
        for eqn in self.stencils:
            dimensions += [i for i in retrieve_dimensions(eqn.lhs) if i not in dimensions]
            dimensions += [i for i in retrieve_dimensions(eqn.rhs) if i not in dimensions]

        # Time dimension is fixed for now
        time_dim = t

        # Derive space dimensions from expression
        self.space_dims = None

        if len(dimensions) > 0:
            self.space_dims = dimensions

            if time_dim in self.space_dims:
                self.space_dims.remove(time_dim)
        else:
            # Default space dimension symbols
            self.space_dims = ((x, z) if len(shape) == 2 else (x, y, z))[:len(shape)]

        # Remove known dimensions from undefined symbols
        for d in dimensions:
            sym_undef.remove(d)

        # TODO: We should check that all undfined symbols have known subs
        # Shift time indices so that LHS writes into t only,
        # eg. u[t+2] = u[t+1] + u[t]  -> u[t] = u[t-1] + u[t-2]
        self.stencils = [eqn.subs(t, t + solve(eqn.lhs.args[0], t)[0])
                         if isinstance(eqn.lhs, TimeData) else eqn
                         for eqn in self.stencils]

        # Convert incoming stencil equations to "indexed access" format
        self.stencils = [Eq(indexify(eqn.lhs), indexify(eqn.rhs))
                         for eqn in self.stencils]

        # Applies CSE
        self.stencils = rewrite(self.stencils, mode=dse)

        # Apply user-defined subs to stencil
        self.stencils = [eqn.subs(subs[0]) for eqn in self.stencils]
        self.propagator = Propagator(self.getName(), nt, shape, self.stencils,
                                     dtype=dtype, spc_border=spc_border,
                                     time_order=time_order, forward=forward,
                                     space_dims=self.space_dims, compiler=self.compiler,
                                     profile=profile, cache_blocking=cache_blocking)
        self.dtype = dtype
        self.nt = nt
        self.shape = shape
        self.spc_border = spc_border
        self.time_order = time_order
        self.symbol_to_data = {}

        for param in self.signature:
            self.propagator.add_devito_param(param)
            self.symbol_to_data[param.name] = param
        self.propagator.stencils = self.stencils

    @property
    def signature(self):
        """List of data object parameters that define the operator signature

        :returns: List of unique input and output data objects
        """
        return self.input_params + [param for param in self.output_params
                                    if param not in self.input_params]

    def apply(self, debug=False):
        """
        :param debug: If True, use Python to apply the operator. Default False.
        :returns: A tuple containing the values of the operator outputs or compiled
                  function and its args
        """
        if debug:
            return self.apply_python()

        self.propagator.run(self.get_args())

        return tuple([param for param in self.output_params])

    def apply_python(self):
        """Uses Python to apply the operator

        :returns: A tuple containing the values of the operator outputs
        """
        self.run_python()

        return tuple([param.data for param in self.output_params])

    def symbol_to_var(self, term, ti, indices=[]):
        """Retrieves the Python data from a symbol

        :param term: The symbol from which the data has to be retrieved
        :param ti: The value of t to use
        :param indices: A list of indices to use for the space dimensions
        :returns: A tuple containing the data and the indices to access it
        """
        arr = self.symbol_to_data[str(term.base.label)].data
        num_ind = []

        for ind in term.indices:
            ind = ind.subs({t: ti}).subs(tuple(zip(self.space_dims, indices)))
            num_ind.append(ind)

        return (arr, tuple(num_ind))

    def run_python(self):
        """
        Execute the operator using Python
        """
        time_loop_limits = self.propagator.time_loop_limits
        time_loop_lambdas_b = tolambda(self.propagator.time_loop_stencils_b)
        time_loop_lambdas_a = tolambda(self.propagator.time_loop_stencils_a)
        stencil_lambdas = tolambda(self.stencils)

        for ti in range(*time_loop_limits):
            # Run time loop stencils before space loop
            for lams, expr in zip(time_loop_lambdas_b,
                                  self.propagator.time_loop_stencils_b):
                lamda = lams[0]
                subs = lams[1]
                arr_lhs, ind_lhs = self.symbol_to_var(expr.lhs, ti)
                args = []

                for sub in subs:
                    arr, ind = self.symbol_to_var(sub, ti)
                    args.append(arr[ind])

                arr_lhs[ind_lhs] = lamda(*args)

            lower_limits = [self.spc_border]*len(self.shape)
            upper_limits = [x-self.spc_border for x in self.shape]
            indices = lower_limits[:]

            # Number of iterations in each dimension
            total_size_arr = [a - b for a, b in zip(upper_limits, lower_limits)]

            # Total number of iterations
            total_iter = reduce(lambda x, y: x*y, total_size_arr)

            # The 2/3 dimensional space loop has been collapsed to a single loop
            for iter_index in range(0, total_iter):
                dimension_limit = 1

                # Calculating 2/3 dimensional index based on 1D index
                indices[0] = lower_limits[0] + iter_index % total_size_arr[0]

                for dimension in range(1, len(self.shape)):
                    dimension_limit *= total_size_arr[dimension]
                    indices[dimension] = int(iter_index / dimension_limit)

                for lams, expr in zip(stencil_lambdas, self.stencils):
                    lamda = lams[0]
                    subs = lams[1]
                    arr_lhs, ind_lhs = self.symbol_to_var(expr.lhs, ti, indices)
                    args = []

                    for s in subs:
                        arr, ind = self.symbol_to_var(s, ti, indices)
                        args.append(arr[ind])

                    arr_lhs[ind_lhs] = lamda(*args)

            # Time loop stencils for after space loop
            for lams, expr in zip(time_loop_lambdas_a,
                                  self.propagator.time_loop_stencils_a):
                lamda = lams[0]
                subs = lams[1]
                arr_lhs, ind_lhs = self.symbol_to_var(expr.lhs, ti)
                args = []

                for s in subs:
                    arr, ind = self.symbol_to_var(s, ti)
                    args.append(arr[ind])

                arr_lhs[ind_lhs] = lamda(*args)

    def getName(self):
        """Gives the name of the class

        :returns: The name of the class
        """
        return self.__class__.__name__

    def get_args(self):
        """
        Initialises all the input args and returns them
        :return: a list of input params
        """
        for param in self.input_params:
            if hasattr(param, 'initialize'):
                param.initialize()
        return [param.data for param in self.signature]


class SimpleOperator(Operator):
    def __init__(self, input_grid, output_grid, kernel, **kwargs):
        assert(input_grid.shape == output_grid.shape)

        nt = input_grid.shape[0]
        shape = input_grid.shape[1:]
        input_params = [input_grid]
        output_params = [output_grid]

        super(SimpleOperator, self).__init__(nt, shape, stencils=kernel,
                                             subs={},
                                             input_params=input_params,
                                             output_params=output_params,
                                             dtype=input_grid.dtype, **kwargs)
