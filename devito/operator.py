from devito.compiler import get_compiler_from_env, IntelMICCompiler
from propagator import Propagator
from sympy import Indexed, lambdify, symbols
import numpy as np
from sympy import Eq, preorder_traversal


__all__ = ['Operator']


def expr_indexify(expr):
    """Convert functions into indexed matrix accesses in sympy expression"""
    replacements = {}
    for e in preorder_traversal(expr):
        if hasattr(e, 'indexed'):
            replacements[e] = e.indexify()
    return expr.xreplace(replacements)


class Operator(object):
    """Class encapsulating a defined operator as defined by the given stencil

    The Operator class is the core abstraction in DeVito that allows
    users to generate high-performance Finite Difference kernels from
    a stencil definition defined from SymPy equations.

    :param subs: SymPy symbols to substitute in the stencil
    :param nt: Number of timesteps to execute
    :param shape: Shape of the data buffer over which to execute
    :param dtype: Data type for the grid buffer
    :param spc_border: Number of spatial padding layers
    :param time_order: Order of the time discretisation
    :param forward: Flag indicating whether to execute forward in time
    :param compiler: Compiler class used to perform JIT compilation.
                     If not provided, the compiler will be inferred from the
                     environment variable DEVITO_ARCH, or default to GNUCompiler.
    :param profile: Flag to enable performance profiling
    :param cache_blocking: Flag to enable cache blocking
    :param block_size: Block size used for cache clocking
    :param stencils: List of (stencil, subs) tuples that define individual
                     stencils and their according substitutions.
    :param input_params: List of symbols that are expected as input.
    :param output_params: List of symbols that define operator output.
    :param factorized: A map given by {string_name:sympy_object} for including factorized terms
    """

    _ENV_VAR_OPENMP = "DEVITO_OPENMP"

    def __init__(self, subs, nt, shape, dtype=np.float32, spc_border=0,
                 time_order=0, forward=True, compiler=None,
                 profile=False, cache_blocking=False, block_size=5,
                 stencils=[], input_params=[], output_params=[], factorized={}):
        # Derive JIT compilation infrastructure
        self.compiler = compiler or get_compiler_from_env()

        # Convert incoming stencil equations to "indexed access" format
        self.stencils = [(Eq(expr_indexify(eqn.lhs), expr_indexify(eqn.rhs)), s)
                         for eqn, s in stencils]
        self.subs = subs
        self.propagator = Propagator(self.getName(), nt, shape, spc_border=spc_border,
                                     time_order=time_order, forward=forward,
                                     compiler=self.compiler, profile=profile,
                                     cache_blocking=cache_blocking, block_size=block_size)
        self.propagator.time_loop_stencils_b = self.propagator.time_loop_stencils_b + getattr(self, "time_loop_stencils_pre", [])
        self.propagator.time_loop_stencils_a = self.propagator.time_loop_stencils_a + getattr(self, "time_loop_stencils_post", [])
        self.params = {}
        self.input_params = input_params
        self.output_params = output_params
        self.dtype = dtype
        self.nt = nt
        self.shape = shape
        self.spc_border = spc_border
        self.symbol_to_data = {}
        for param in self.input_params + self.output_params:
            self.params[param.name] = param
            self.propagator.add_devito_param(param)
            self.symbol_to_data[param.name] = param
        self.propagator.subs = self.subs
        self.propagator.stencils, self.propagator.stencil_args = zip(*self.stencils)
        self.propagator.factorized = factorized
    

    def apply(self, debug=False):
        if debug:
            return self.apply_python()
        f = self.propagator.cfunction
        for param in self.input_params:
            param.initialize()
        args = [param.data for param in self.input_params + self.output_params]
        if isinstance(self.compiler, IntelMICCompiler):
            # Off-load propagator kernel via pymic stream
            self.compiler._stream.invoke(f, *args)
            self.compiler._stream.sync()
        else:
            f(*args)
        return tuple([param.data for param in self.output_params])

    def apply_python(self):
        self.run_python()
        return tuple([param.data for param in self.output_params])

    def find_free_terms(self, expr):
        free_terms = []
        for term in expr.args:
            if isinstance(term, Indexed):
                free_terms.append(term)
            else:
                free_terms += self.find_free_terms(term)
        return free_terms

    def symbol_to_var(self, term, ti, indices=[]):
        arr = self.symbol_to_data[str(term.base.label)].data
        if len(self.shape) == 2:
            space_dims = symbols("x z")
        else:
            space_dims = symbols("x y z")
        num_ind = []
        for ind in term.indices:
            ind = ind.subs({symbols("t"): ti}).subs(tuple(zip(space_dims, indices)))
            num_ind.append(ind)
        return (arr, tuple(num_ind))

    def expr_to_lambda(self, expr_arr):
        lambdas = []
        for expr in expr_arr:
            terms = self.find_free_terms(expr.rhs)
            term_symbols = [symbols("i%d" % i) for i in range(len(terms))]
            # Substitute IndexedBase references to simple variables
            # lambdify doesn't support IndexedBase references in expressions
            expr_to_lambda = expr.rhs.subs(dict(zip(terms, term_symbols)))
            lambdified = lambdify(term_symbols, expr_to_lambda)
            lambdas.append((lambdified, terms))
        return lambdas

    def run_python(self):
        time_loop_limits = self.propagator.time_loop_limits
        time_loop_lambdas_b = self.expr_to_lambda(self.propagator.time_loop_stencils_b)
        time_loop_lambdas_a = self.expr_to_lambda(self.propagator.time_loop_stencils_a)
        stencils = [stencil.subs(dict(zip(self.subs, args))) for stencil, args in self.stencils]
        stencil_lambdas = self.expr_to_lambda(stencils)
        for ti in range(*time_loop_limits):
            # Run time loop stencils before space loop
            for lams, expr in zip(time_loop_lambdas_b, self.propagator.time_loop_stencils_b):
                lamda = lams[0]
                subs = lams[1]
                arr_lhs, ind_lhs = self.symbol_to_var(expr.lhs, ti)
                args = []
                for x in subs:
                    arr, ind = self.symbol_to_var(x, ti)
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
                    arr_lhs, ind_lhs = self.symbol_to_var(expr[0].lhs, ti, indices)
                    args = []
                    for x in subs:
                        arr, ind = self.symbol_to_var(x, ti, indices)
                        args.append(arr[ind])
                    arr_lhs[ind_lhs] = lamda(*args)
            # Time loop stencils for after space loop
            for lams, expr in zip(time_loop_lambdas_a, self.propagator.time_loop_stencils_a):
                lamda = lams[0]
                subs = lams[1]
                arr_lhs, ind_lhs = self.symbol_to_var(expr.lhs, ti)
                args = []
                for x in subs:
                    arr, ind = self.symbol_to_var(x, ti)
                    args.append(arr[ind])
                arr_lhs[ind_lhs] = lamda(*args)

    def getName(self):
        return self.__class__.__name__


class SimpleOperator(Operator):
    def __init__(self, input_grid, output_grid, kernel, **kwargs):
        assert(input_grid.shape == output_grid.shape)
        nt = input_grid.shape[0]
        shape = input_grid.shape[1:]
        input_params = [input_grid, output_grid]
        output_params = []
        stencils = zip(kernel, [[]]*len(kernel))
        super(SimpleOperator, self).__init__([], nt, shape, stencils=stencils,
                                             input_params=input_params,
                                             output_params=output_params,
                                             dtype=input_grid.dtype, **kwargs)
