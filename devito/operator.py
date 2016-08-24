import numpy as np
from sympy import (Add, Eq, Function, Indexed, IndexedBase, Symbol, cse,
                   lambdify, preorder_traversal, solve, symbols)
from sympy.abc import t
from sympy.utilities.iterables import numbered_symbols

from devito.compiler import get_compiler_from_env
from devito.interfaces import SymbolicData, TimeData
from devito.propagator import Propagator

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
    """Convert functions into indexed matrix accesses in sympy expression

    :param expr: SymPy function expression to be converted
    """
    replacements = {}

    for e in preorder_traversal(expr):
        if hasattr(e, 'indexed'):
            replacements[e] = e.indexify()

    return expr.xreplace(replacements)


def expr_cse(expr):
    """Performs common subexpression elimination on expressio

    :param expr: Sympy expression on which CSE needs to be performed
    """
    expr = expr if isinstance(expr, list) else [expr]

    temps, stencils = cse(expr, numbered_symbols("temp"))

    # Restores the LHS
    for i in range(len(expr)):
        stencils[i] = Eq(expr[i].lhs, stencils[i].rhs)

    to_revert = {}
    to_keep = []

    # Restores IndexedBases if they are collected by CSE and
    # reverts changes to simple index operations (eg: t - 1)
    for temp, value in temps:
        if isinstance(value, IndexedBase):
            to_revert[temp] = value
        elif isinstance(value, Add):
            to_revert[temp] = value
        else:
            to_keep.append((temp, value))

    # Restores the IndexedBases in the assignments to revert
    for temp, value in to_revert.items():
        s_dict = {}
        for arg in preorder_traversal(value):
            if isinstance(value, Indexed):
                if value.base.label in to_revert:
                    s_dict[arg] = Indexed(to_revert[value.base.label], *value.indices)
        to_revert[temp] = value.xreplace(s_dict)

    subs_dict = {}

    # Builds a dictionary of the replacements
    for expr in stencils + [assign for temp, assign in to_keep]:
        for arg in preorder_traversal(expr):
            if isinstance(arg, Indexed):
                new_indices = []
                for index in arg.indices:
                    if index in to_revert:
                        new_indices.append(to_revert[index])
                    else:
                        new_indices.append(index)
                if arg.base.label in to_revert:
                    subs_dict[arg] = Indexed(to_revert[arg.base.label], *new_indices)
                elif tuple(new_indices) != arg.indices:
                    subs_dict[arg] = Indexed(arg.base, *new_indices)
            if arg in to_revert:
                subs_dict[arg] = to_revert[arg]

    stencils = [stencil.xreplace(subs_dict) for stencil in stencils]

    for i in range(len(to_keep)):
        to_keep[i] = Eq(to_keep[i][0], to_keep[i][1].xreplace(subs_dict))

    return to_keep + stencils


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
    :param cache_blocking: Flag to enable cache blocking
    :param block_size: Block size used for cache clocking. Can be either a single number
                       used for all dimensions or a list stating block sizes for each
                       dimension. Set block size to None to skip blocking on that dim
    :param input_params: List of symbols that are expected as input.
    :param output_params: List of symbols that define operator output.
    :param factorized: A map given by {string_name:sympy_object} for including factorized
                       terms
    """

    def __init__(self, nt, shape, dtype=np.float32, stencils=[],
                 subs=[], spc_border=0, time_order=0,
                 forward=True, compiler=None, profile=False,
                 cache_blocking=False, block_size=5,
                 input_params=None, output_params=None, factorized={}):
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
            lhs_def, lhs_undef = expr_symbols(eqn.lhs)
            sym_undef.update(lhs_undef)

            if self.output_params is None:
                self.output_params = list(lhs_def)

            rhs_def, rhs_undef = expr_symbols(eqn.rhs)
            sym_undef.update(rhs_undef)

            if self.input_params is None:
                self.input_params = list(rhs_def)

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
        self.stencils = [Eq(expr_indexify(eqn.lhs), expr_indexify(eqn.rhs))
                         for eqn in self.stencils]

        for name, value in factorized.items():
            factorized[name] = expr_indexify(value)

        # Apply user-defined substitutions to stencil
        self.stencils = [eqn.subs(subs[0]) for eqn in self.stencils]

        # Applies CSE
        self.stencils = expr_cse(self.stencils)

        self.propagator = Propagator(
            self.getName(), nt, shape, self.stencils, factorized=factorized, dtype=dtype,
            spc_border=spc_border, time_order=time_order, forward=forward,
            space_dims=self.space_dims, compiler=self.compiler, profile=profile,
            cache_blocking=cache_blocking, block_size=block_size
        )
        self.dtype = dtype
        self.nt = nt
        self.shape = shape
        self.spc_border = spc_border
        self.symbol_to_data = {}
        for param in self.signature:
            self.propagator.add_devito_param(param)
            self.symbol_to_data[param.name] = param
        self.propagator.stencils = self.stencils
        self.propagator.factorized = factorized
        for name, val in factorized.items():
            if forward:
                self.propagator.factorized[name] = \
                    expr_indexify(val.subs(t, t - 1)).subs(subs[1])
            else:
                self.propagator.factorized[name] = \
                    expr_indexify(val.subs(t, t + 1)).subs(subs[1])

    @property
    def signature(self):
        """List of data object parameters that define the operator signature

        :returns: List of unique input and output data objects
        """
        return self.input_params + [param for param in self.output_params
                                    if param not in self.input_params]

    def apply(self, debug=False):
        """:param debug: If True, use Python to apply the operator. Default False.

        :returns: A tuple containing the values of the operator outputs
        """
        if debug:
            return self.apply_python()

        for param in self.input_params:
            if hasattr(param, 'initialize'):
                param.initialize()

        args = [param.data for param in self.signature]
        self.propagator.run(args)

        return tuple([param for param in self.output_params])

    def apply_python(self):
        """Uses Python to apply the operator

        :returns: A tuple containing the values of the operator outputs
        """
        self.run_python()

        return tuple([param.data for param in self.output_params])

    def find_free_terms(self, expr):
        """Finds free terms in an expression

        :param expr: The expression to search
        :returns: A list of free terms in the expression
        """
        free_terms = []

        for term in expr.args:
            if isinstance(term, Indexed):
                free_terms.append(term)
            else:
                free_terms += self.find_free_terms(term)

        return free_terms

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
            ind = ind.subs({symbols("t"): ti}).subs(tuple(zip(self.space_dims, indices)))
            num_ind.append(ind)

        return (arr, tuple(num_ind))

    def expr_to_lambda(self, expr_arr):
        """Tranforms a list of expressions in a list of lambdas

        :param expr_arr: The list of expressions to be transformed
        :returns: The list of lambdas resulting from the expressions
        """
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
        """Executes the operator using Python
        """
        time_loop_limits = self.propagator.time_loop_limits
        time_loop_lambdas_b = self.expr_to_lambda(self.propagator.time_loop_stencils_b)
        time_loop_lambdas_a = self.expr_to_lambda(self.propagator.time_loop_stencils_a)
        stencil_lambdas = self.expr_to_lambda(self.stencils)

        for ti in range(*time_loop_limits):
            # Run time loop stencils before space loop
            for lams, expr in zip(time_loop_lambdas_b,
                                  self.propagator.time_loop_stencils_b):
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
            for lams, expr in zip(time_loop_lambdas_a,
                                  self.propagator.time_loop_stencils_a):
                lamda = lams[0]
                subs = lams[1]
                arr_lhs, ind_lhs = self.symbol_to_var(expr.lhs, ti)
                args = []

                for x in subs:
                    arr, ind = self.symbol_to_var(x, ti)
                    args.append(arr[ind])

                arr_lhs[ind_lhs] = lamda(*args)

    def getName(self):
        """Gives the name of the class

        :returns: The name of the class
        """
        return self.__class__.__name__


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
