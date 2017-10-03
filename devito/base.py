"""Metaclasses used to construct classes of proper backend type at runtime."""

from __future__ import absolute_import

from sympy.core.compatibility import with_metaclass

from devito.backends import _BackendSelector
import devito.interfaces as interfaces
import devito.pointdata as pointdata
import devito.operator as operator


class Scalar(with_metaclass(_BackendSelector, interfaces.Scalar)):
    pass


class TensorFunction(with_metaclass(_BackendSelector, interfaces.TensorFunction)):
    pass


class ConstantData(with_metaclass(_BackendSelector, interfaces.ConstantData)):
    pass


class Function(with_metaclass(_BackendSelector, interfaces.Function)):
    pass


class TimeFunction(with_metaclass(_BackendSelector, interfaces.TimeFunction)):
    pass


class PointData(with_metaclass(_BackendSelector, pointdata.PointData)):
    pass


class Operator(with_metaclass(_BackendSelector, operator.Operator)):
    pass
