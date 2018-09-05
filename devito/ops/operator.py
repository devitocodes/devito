from devito.ir.support import align_accesses
from devito.operator import OperatorRunnable

__all__ = ['Operator']


class OperatorOPS(OperatorRunnable):

    def _specialize_exprs(self, expressions):
        # Align data accesses to the computational domain
        key = lambda i: i.is_TensorFunction
        expressions = [align_accesses(e, key=key) for e in expressions]
        return super(OperatorOPS, self)._specialize_exprs(expressions)

class Operator(object):

    def __new__(cls, *args, **kwargs):
        cls = OperatorOPS
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        return obj
