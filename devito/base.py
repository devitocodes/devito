"""Metaclasses used to construct classes of proper backend type at runtime."""

from __future__ import absolute_import

from sympy.core.compatibility import with_metaclass

from devito.backends import _BackendSelector
import devito.types as types
import devito.function as function
import devito.operator as operator


class Scalar(with_metaclass(_BackendSelector, types.Scalar)):
    pass


class Array(with_metaclass(_BackendSelector, types.Array)):
    pass


class Constant(with_metaclass(_BackendSelector, function.Constant)):
    pass


class Function(with_metaclass(_BackendSelector, function.Function)):
    pass


class TimeFunction(with_metaclass(_BackendSelector, function.TimeFunction)):
    pass


class SparseFunction(with_metaclass(_BackendSelector, function.SparseFunction)):
    pass


class SparseTimeFunction(with_metaclass(_BackendSelector, function.SparseTimeFunction)):
    pass


class Operator(with_metaclass(_BackendSelector, operator.Operator)):
    pass


class CacheManager(with_metaclass(_BackendSelector, types.CacheManager)):
    pass
