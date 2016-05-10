from jit_manager import JitManager
from propagator import Propagator


class Operator(object):
    def __init__(self, nt, shape):
        self.propagator = Propagator(self.getName(), nt, shape, spc_border=1, time_order=2)

    def get_callable(self):
        prop = self.get_propagator()
        self.jit_manager = JitManager([prop], dtype=self.dtype)
        return self.jit_manager.get_wrapped_functions()[0]

    def get_propagator(self):
        return self._prepare()

    def getName(self):
        return self.__class__.__name__
