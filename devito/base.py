"""Metaclasses used to construct classes of proper backend type at runtime."""

from __future__ import absolute_import

from sympy.core.compatibility import with_metaclass

from devito.backends import _BackendSelector
import devito.interfaces as interfaces
import devito.pointdata as pointdata
import devito.operator as operator


class ScalarFunction(with_metaclass(_BackendSelector, interfaces.ScalarFunction)):
    pass


class TensorFunction(with_metaclass(_BackendSelector, interfaces.TensorFunction)):
    pass


class DenseData(with_metaclass(_BackendSelector, interfaces.DenseData)):
    pass


class TimeData(with_metaclass(_BackendSelector, interfaces.TimeData)):
    pass


class PointData(with_metaclass(_BackendSelector, pointdata.PointData)):
    pass


class Operator(with_metaclass(_BackendSelector, operator.Operator)):
    pass
