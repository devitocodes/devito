from jit_manager import JitManager
from propagator import Propagator
from sympy import Indexed, lambdify, symbols
from sympy.utilities.autowrap import autowrap
import os


class Operator(object):
    _ENV_VAR_OPENMP = "DEVITO_OPENMP"

    def __init__(self, subs, nt, shape, dtype, spc_border=0, time_order=0, forward=True, profile=False, cache_blocking=False, block_size=5):
        self.subs = subs
        self.openmp = os.environ.get(self._ENV_VAR_OPENMP) == "1"
        self.propagator = Propagator(self.getName(), nt, shape, spc_border, forward, time_order, self.openmp, profile, cache_blocking, block_size)
        self.propagator.time_loop_stencils_b = self.propagator.time_loop_stencils_b + getattr(self, "time_loop_stencils_pre", [])
        self.propagator.time_loop_stencils_a = self.propagator.time_loop_stencils_a + getattr(self, "time_loop_stencils_post", [])
        self.params = {}
        self.dtype = dtype
        self.nt = nt
        self.shape = shape
        self.spc_border = spc_border
        for param in self.input_params:
            self.params[param.name] = param
            self.propagator.add_devito_param(param)
        for param in self.output_params:
            self.params[param.label] = param
            self.propagator.add_devito_param(param)
        self.symbol_to_data = {}
        for param in self.input_params+self.output_params:
            self.symbol_to_data[param.name]=param
        self.propagator.subs = self.subs
        self.propagator.stencils, self.propagator.stencil_args = zip(*self.stencils)

    def apply(self, debug=False):
        if debug:
            return self.apply_python()
        f = self.get_callable()
        for param in self.input_params:
            param.initialize()
        args = [param.data for param in self.input_params + self.output_params]
        f(*args)
        return tuple([param.data for param in self.output_params])

    def apply_python(self):
        self.run_python()

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
        if len(self.shape)==2:
            space_dims = symbols("x z")
        else:
            space_dims = symbols("x y z")
        num_ind = []
        for ind in term.indices:
            ind = ind.subs({symbols("t"):ti}).subs(tuple(zip(space_dims, indices)))
            num_ind.append(ind)
        return arr[tuple(num_ind)]

    def expr_to_lambda(self, expr_arr):
        lambdas = []
        for expr in expr_arr:
            terms = self.find_free_terms(expr.rhs)
            term_symbols = [symbols("i%d"%i) for i in range(len(terms))]
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
        stencils = []
        for stencil, args in self.stencils:
            stencil = stencil.subs(dict(zip(self.subs, args)))
            stencils.append(stencil)
        stencil_lambdas = self.expr_to_lambda(stencils)
        symbol_t = symbols("t")
        if len(self.shape)==2:
            space_dims = symbols("x z")
        else:
            space_dims = symbols("x y z")
        for ti in range(*time_loop_limits):
            # Run time loop stencils before space loop
            for lams, expr in zip(time_loop_lambdas_b, self.propagator.time_loop_stencils_b):
                lamda = lams[0]
                subs = lams[1]
                lhs = self.symbol_to_var(expr.lhs, ti)
                args = [self.symbol_to_var(x, ti, indices) for x in subs]
                lhs = lamda(*args)
            lower_limits = [self.spc_border]*len(self.shape)
            upper_limits = [x-self.spc_border for x in self.shape]
            indices = lower_limits[:]
            # Number of iterations in each dimension
            total_size_arr = [a - b for a, b in zip(upper_limits, lower_limits)]
            # Total number of iterations
            total_iter = reduce(lambda x, y:x*y,total_size_arr)
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
                    lhs = self.symbol_to_var(expr[0].lhs, ti, indices)
                    args = [self.symbol_to_var(x, ti, indices) for x in subs]
                    lhs = lamda(*args)
            # Time loop stencils for after space loop
            for lams, expr in zip(time_loop_lambdas_a, self.propagator.time_loop_stencils_a):
                lamda = lams[0]
                subs = lams[1]
                lhs = self.symbol_to_var(expr.lhs, ti)
                args = [self.symbol_to_var(x, ti, indices) for x in subs]
                lhs = lamda(*args)

    def get_callable(self):
        self.jit_manager = JitManager([self.propagator], dtype=self.dtype, openmp=self.openmp)
        return self.jit_manager.get_wrapped_functions()[0]

    def getName(self):
        return self.__class__.__name__


class SimpleOperator(Operator):
    def __init__(self, input_grid, output_grid, kernel, **kwargs):
        assert(input_grid.shape == output_grid.shape)
        nt = input_grid.shape[0]
        shape = input_grid.shape[1:]
        self.input_params = [input_grid, output_grid]
        self.output_params = []
        self.stencils = zip(kernel, [[]]*len(kernel))
        super(SimpleOperator, self).__init__([], nt, shape, input_grid.dtype, **kwargs)
