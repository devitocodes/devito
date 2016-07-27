from devito.compiler import get_compiler_from_env, IntelMICCompiler
from devito.interfaces import SymbolicData, TimeData
from devito.propagator import Propagator
from sympy import Indexed, lambdify, symbols, solve
from sympy import Eq, Function, Symbol, preorder_traversal
from sympy.abc import t
import numpy as np


__all__ = ['Operator']


def expr_dimensions(expr):
    """Collects all function dimensions used in a sympy expression"""
    dimensions = []
    for e in preorder_traversal(expr):
        if isinstance(e, TimeData):
            dimensions += e.indices(e.shape[1:])
        elif isinstance(e, SymbolicData):
            dimensions += e.indices(e.shape)
    return list(set(dimensions))


def expr_symbols(expr):
    """Collects defined and undefined symbols used in a sympy expression

    Defined symbols are functions that have an associated :class
    SymbolicData: object, or dimensions that are known to the devito
    engine. Undefined symbols are generic `sympy.Function` or
    `sympy.Symbol` objects that need to be substituted before generating
    operator C code.
    """
    defined = set()
    undefined = set()
    for e in preorder_traversal(expr):
        if isinstance(e, TimeData):
            defined.add(e.func(*e.indices(e.shape[1:])))
        elif isinstance(e, SymbolicData):
            defined.add(e.func(*e.indices(e.shape)))
        elif isinstance(e, Function):
            undefined.add(e)
        elif isinstance(e, Symbol):
            undefined.add(e)
    return list(defined), list(undefined)


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

    :param nt: Number of timesteps to execute
    :param shape: Shape of the data buffer over which to execute
    :param dtype: Data type for the grid buffer
    :param stencils: SymPy equation or list of equations that define the
                     stencil used to create the kernel of this Operator.
    :param substitutions: Dict or list of dicts containing the SymPy symbol
                          substitutions for each stencil respectively.
    :param spc_border: Number of spatial padding layers
    :param time_order: Order of the time discretisation
    :param forward: Flag indicating whether to execute forward in time
    :param compiler: Compiler class used to perform JIT compilation.
                     If not provided, the compiler will be inferred from the
                     environment variable DEVITO_ARCH, or default to GNUCompiler.
    :param profile: Flag to enable performance profiling
    :param cache_blocking: Flag to enable cache blocking
    :param block_size: Block size used for cache clocking
    :param input_params: List of symbols that are expected as input.
    :param output_params: List of symbols that define operator output.
    """

    _ENV_VAR_OPENMP = "DEVITO_OPENMP"

    def __init__(self, nt, shape, dtype=np.float32, stencils=[],
                 substitutions=[], spc_border=0, time_order=0,
                 forward=True, compiler=None, profile=False,
                 cache_blocking=False, block_size=5,
                 input_params=None, output_params=None):
        # Derive JIT compilation infrastructure
        self.compiler = compiler or get_compiler_from_env()

        # Ensure stencil and substititutions are lists internally
        self.stencils = stencils if isinstance(stencils, list) else [stencils]
        substitutions = substitutions if isinstance(substitutions, list) else [substitutions]
        self.input_params = input_params
        self.output_params = output_params
        # Pull all dimension indices from the incoming stencil
        dimensions = set()
        for eqn in self.stencils:
            dimensions.update(set(expr_dimensions(eqn.lhs)))
            dimensions.update(set(expr_dimensions(eqn.rhs)))
        # Time dimension is fixed for now
        time_dim = symbols("t")
        # Derive space dimensions from expression
        self.space_dims = None
        if len(dimensions) > 0:
            self.space_dims = list(dimensions)
            if time_dim in self.space_dims:
                self.space_dims.remove(time_dim)
        else:
            # Default space dimension symbols
            self.space_dims = symbols("x z" if len(shape) == 2 else "x y z")[:len(shape)]
        # Get functions and symbols in LHS/RHS and update params
        sym_undef = set()
        for eqn in self.stencils:
            lhs_def, lhs_undef = expr_symbols(eqn.lhs)
            sym_undef.update(lhs_undef)
            if self.output_params is None:
                self.output_params = list(lhs_def)
            rhs_def, rhs_undef = expr_symbols(eqn.rhs)
            sym_undef.update(rhs_undef)
            if self.input_params is None:
                self.input_params = list(rhs_def)
        # Remove output from input params
        for p in self.input_params:
            if p in self.output_params:
                self.output_params.remove(p)
        # Remove known dimensions from undefined symbols
        for d in dimensions:
            sym_undef.remove(d)
        # TODO: We should check that all undfined symbols have known substitutions

        # Shift time indices so that LHS writes into t only,
        # eg. u[t+2] = u[t+1] + u[t]  -> u[t] = u[t-1] + u[t-2]
        self.stencils = [eqn.subs(t, t + solve(eqn.lhs.args[0], t)[0])
                         if isinstance(eqn.lhs, TimeData) else eqn
                         for eqn in self.stencils]
        # Convert incoming stencil equations to "indexed access" format
        self.stencils = [Eq(expr_indexify(eqn.lhs), expr_indexify(eqn.rhs))
                         for eqn in self.stencils]
        # Apply user-defined substitutions to stencil
        self.stencils = [eqn.subs(args) for eqn, args in zip(self.stencils, substitutions)]
        self.propagator = Propagator(self.getName(), nt, shape, spc_border=spc_border,
                                     time_order=time_order, forward=forward,
                                     space_dims=self.space_dims,
                                     compiler=self.compiler, profile=profile,
                                     cache_blocking=cache_blocking, block_size=block_size)
        self.dtype = dtype
        self.nt = nt
        self.shape = shape
        self.spc_border = spc_border
        self.symbol_to_data = {}
        for param in self.input_params + self.output_params:
            self.propagator.add_devito_param(param)
            self.symbol_to_data[param.name] = param
        self.propagator.stencils = self.stencils

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
        num_ind = []
        for ind in term.indices:
            ind = ind.subs({symbols("t"): ti}).subs(tuple(zip(self.space_dims, indices)))
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
        stencil_lambdas = self.expr_to_lambda(self.stencils)
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
                    arr_lhs, ind_lhs = self.symbol_to_var(expr.lhs, ti, indices)
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
        input_params = [input_grid]
        output_params = [output_grid]
        super(SimpleOperator, self).__init__(nt, shape, stencils=kernel,
                                             substitutions={},
                                             input_params=input_params,
                                             output_params=output_params,
                                             dtype=input_grid.dtype, **kwargs)
