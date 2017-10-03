"""Metaclasses used to construct classes of proper backend type at runtime."""

from __future__ import absolute_import

from sympy.core.compatibility import with_metaclass

from devito.backends import _BackendSelector
import devito.interfaces as interfaces
import devito.sparsefunction as sparsefunction
import devito.operator as operator


class Scalar(with_metaclass(_BackendSelector, interfaces.Scalar)):
    pass


class Array(with_metaclass(_BackendSelector, interfaces.Array)):
    pass


class Constant(with_metaclass(_BackendSelector, interfaces.Constant)):
    pass


class Function(with_metaclass(_BackendSelector, interfaces.Function)):
    pass


class TimeFunction(with_metaclass(_BackendSelector, interfaces.TimeFunction)):
    pass


class SparseFunction(with_metaclass(_BackendSelector, sparsefunction.SparseFunction)):
    pass


class Operator(with_metaclass(_BackendSelector, operator.Operator)):
    pass
