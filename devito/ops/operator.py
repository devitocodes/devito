from devito.operator import Operator

from devito.ops import ops_configuration

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    _default_headers = Operator._default_headers + ['#define restrict __restrict']

    def __new__(cls, expressions, **kwargs):
        op = super(OperatorOPS, cls).__new__(cls, expressions, **kwargs)

        op._compiler = ops_configuration['compiler'].copy()

        return op

    @property
    def hcode(self):
        return ''.join(str(kernel.root) for kernel in self._func_table.values())

    def _compile(self):
        self._includes.append('%s.h' % self._soname)
        if self._lib is None:
            self._compiler.jit_compile(self._soname, str(self.ccode), str(self.hcode))
