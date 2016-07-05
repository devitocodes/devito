from jit_manager import JitManager
from propagator import Propagator
from at_controller import AtController
import os


class Operator(object):
    _ENV_VAR_OPENMP = "DEVITO_OPENMP"

    def __init__(self, subs, nt, shape, dtype, spc_border=0, time_order=0, forward=True, profile=False,
                 cache_blocking=False, block_size=5, auto_tune=False):
        self.subs = subs
        self.cache_blocking = cache_blocking
        self.auto_tune = auto_tune
        self.spc_border = spc_border
        self.time_order = time_order
        self.openmp = os.environ.get(self._ENV_VAR_OPENMP) == "1"
        self.propagator = Propagator(self.getName(), nt, shape, self.spc_border, forward, self.time_order, self.openmp,
                                     profile, self.cache_blocking, block_size, self.auto_tune)
        self.propagator.time_loop_stencils_b = self.propagator.time_loop_stencils_b + getattr(self, "time_loop_stencils_pre", [])
        self.propagator.time_loop_stencils_a = self.propagator.time_loop_stencils_a + getattr(self, "time_loop_stencils_post", [])
        self.params = {}
        self.dtype = dtype
        for param in self.input_params:
            self.params[param.name] = param
            self.propagator.add_devito_param(param)
        for param in self.output_params:
            self.params[param.name] = param
            self.propagator.add_devito_param(param)
        self.propagator.subs = self.subs
        self.propagator.stencils, self.propagator.stencil_args = zip(*self.stencils)

    def apply(self):
        f = self.get_callable()
        for param in self.input_params:
            param.initialize()
        args = [param.data for param in self.input_params + self.output_params]
        f(*args)
        return tuple([param.data for param in self.output_params])

    def get_callable(self):           # each propagator passed here will add a function to generated cpp script
        self.jit_manager = JitManager([self.propagator], dtype=self.dtype, openmp=self.openmp)
        wrapped_function = self.jit_manager.get_wrapped_functions()[0]

        if self.auto_tune and self.cache_blocking:
            at_controller = AtController()                                      # = space order
            at_controller.auto_tune(self.jit_manager.src_file, self.time_order, self.spc_border * 2)

        return wrapped_function

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
