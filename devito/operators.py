from jit_manager import JitManager
from propagator import Propagator


class Operator(object):
    def __init__(self, subs, nt, shape, spc_border, time_order, forward, dtype):
        self.subs = subs
        self.propagator = Propagator(self.getName(), nt, shape, spc_border, forward, time_order)
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
        self.propagator.enable_oi(True, dtype)

    def apply(self):
        for param in self.input_params:
            param.initialize()
        f = self.get_callable()
        args = [param.data for param in self.input_params + self.output_params]
        f(*args)
        return tuple([param.data for param in self.output_params])

    def get_callable(self):
        self.jit_manager = JitManager([self.propagator], dtype=self.dtype)
        return self.jit_manager.get_wrapped_functions()[0]

    def getName(self):
        return self.__class__.__name__
