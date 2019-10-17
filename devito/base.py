"""Metaclasses used to construct classes of proper backend type at runtime."""

from sympy.core.compatibility import with_metaclass

from devito.backends import _BackendSelector
import devito.types.basic as basic
import devito.types.caching as caching
import devito.types.dense as dense
import devito.types.grid as grid
import devito.types.sparse as sparse
import devito.types.constant as constant
import devito.operator as operator


class Scalar(with_metaclass(_BackendSelector, basic.Scalar)):
    pass


class Array(with_metaclass(_BackendSelector, basic.Array)):
    pass


class Constant(with_metaclass(_BackendSelector, constant.Constant)):
    pass


class Function(with_metaclass(_BackendSelector, dense.Function)):
    pass


class TimeFunction(with_metaclass(_BackendSelector, dense.TimeFunction)):
    pass


class SparseFunction(with_metaclass(_BackendSelector, sparse.SparseFunction)):
    pass


class SparseTimeFunction(with_metaclass(_BackendSelector, sparse.SparseTimeFunction)):
    pass


class PrecomputedSparseFunction(with_metaclass(_BackendSelector, sparse.PrecomputedSparseFunction)):  # noqa
    pass


class PrecomputedSparseTimeFunction(with_metaclass(_BackendSelector, sparse.PrecomputedSparseTimeFunction)):  # noqa
    pass


class Grid(with_metaclass(_BackendSelector, grid.Grid)):
    pass


class Operator(with_metaclass(_BackendSelector, operator.Operator)):
    pass


class CacheManager(with_metaclass(_BackendSelector, caching.CacheManager)):
    pass
