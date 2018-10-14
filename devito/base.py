"""Metaclasses used to construct classes of proper backend type at runtime."""

from __future__ import absolute_import

from sympy.core.compatibility import with_metaclass

from devito.backends import _BackendSelector
from devito.grid import Grid
import devito.functions.types as types
import devito.functions.dense as function
import devito.functions.sparse as sparse
import devito.functions.constants as constant
import devito.operator as operator


class Scalar(with_metaclass(_BackendSelector, types.Scalar)):
    pass


class Array(with_metaclass(_BackendSelector, types.Array)):
    pass


class Constant(with_metaclass(_BackendSelector, constant.Constant)):
    pass


class Function(with_metaclass(_BackendSelector, function.Function)):
    pass


class TimeFunction(with_metaclass(_BackendSelector, function.TimeFunction)):
    pass


class SparseFunction(with_metaclass(_BackendSelector, sparse.SparseFunction)):
    pass


class SparseTimeFunction(with_metaclass(_BackendSelector, sparse.SparseTimeFunction)):
    pass


class PrecomputedSparseFunction(with_metaclass(_BackendSelector, sparse.PrecomputedSparseFunction)):  # noqa
    pass


class PrecomputedSparseTimeFunction(with_metaclass(_BackendSelector, sparse.PrecomputedSparseTimeFunction)):  # noqa
    pass


class Grid(with_metaclass(_BackendSelector, Grid)):
    pass


class Operator(with_metaclass(_BackendSelector, operator.Operator)):
    pass


class CacheManager(with_metaclass(_BackendSelector, types.CacheManager)):
    pass
