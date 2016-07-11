from jit_manager import JitManager
from propagator import Propagator
from at_controller import AtController
import os
import numpy as np


__all__ = ['Operator']


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
    :param profile: Flag to enable performance profiling
    :param cache_blocking: Flag to enable cache blocking
    :param block_size: Block size used for cache clocking
    :param auto_tune: Use Intel ISAT to auto-tune block size
    :param stencils: List of (stencil, subs) tuples that define individual
                     stencils and their according substitutions.
    :param input_params: List of symbols that are expected as input.
    :param output_params: List of symbols that define operator output.
    """

    _ENV_VAR_OPENMP = "DEVITO_OPENMP"

    def __init__(self, subs, nt, shape, dtype=np.float32, spc_border=1,
                 time_order=1, forward=True, profile=False,
                 cache_blocking=False, block_size=5, auto_tune=False,
                 stencils=[], input_params=[], output_params=[]):
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
        self.stencils = stencils
        self.input_params = input_params
        self.output_params = output_params
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
            #                               current isat insall dir. Change based on your environment
            at_controller = AtController("%s/isat" % os.getenv("HOME"))         # = space order
            at_controller.auto_tune(self.jit_manager.src_file, self.time_order, self.spc_border * 2)

        return wrapped_function

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
