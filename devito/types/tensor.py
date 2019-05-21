from collections import OrderedDict

import sympy
import numpy as np
from sympy.core.sympify import converter as sympify_converter

from cached_property import cached_property

from devito.finite_differences import Differentiable, generate_fd_shortcuts
from devito.logger import error
from devito.tools.abc import Evaluable
from devito.types.utils import NODE
from devito.types.basic import AbstractCachedTensor
from devito.types.dense import Function, TimeFunction

__all__ = ['TensorFunction', 'TensorTimeFunction', 'VectorFunction', 'VectorTimeFunction']



class TensorFunction(AbstractCachedTensor, Differentiable):
    """
    """
    _sub_type = Function
    _op_priority = Differentiable._op_priority + 1.
    _class_priority = 10

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self._is_symmetric = kwargs.get('symmetric', True)
            self._is_diagonal = kwargs.get('diagonal', False)
            self._staggered = kwargs.get('staggered', self.space_dimensions)
            self._grid = kwargs.get('grid')
            self._space_order = kwargs.get('space_order', 1)

    @classmethod
    def __setup_subfunc__(cls, *args, **kwargs):
        comps = kwargs.get("components")
        if comps is not None:
            return comps
        funcs = []
        dims = kwargs.get("grid").dimensions
        stagg = kwargs.get("staggered", dims)
        name = kwargs.get("name")
        symm = kwargs.get('symmetric', True)
        # Fill tensor, only upper diagonal if symmetric
        for i, d in enumerate(dims):
            start = i+1 if symm else 0
            funcs2 = [0 for _ in range(i+1)] if symm else []
            for j in range(start, len(dims)):
                kwargs["name"] = name+"_%s%s"%(d.name, dims[j].name)
                kwargs["staggered"] = NODE if i == j else (d, dims[j]) 
                funcs2.append(cls._sub_type(**kwargs))
            funcs.append(funcs2)
        
        # Symmetrize and fill diagonal if symmetric
        if symm:
            funcs = np.array(funcs) + np.array(funcs).T
            for i in range(len(dims)):
                kwargs["name"] = name+"_%s%s"%(dims[i].name, dims[i].name)
                kwargs["staggered"] = NODE
                funcs[i, i] = cls._sub_type(**kwargs)
            funcs = funcs.tolist()
        return funcs
      
    def __getattr__(self, name):
        """
        Try calling a dynamically created FD shortcut.

        Notes
        -----
        This method acts as a fallback for __getattribute__
        """
        if name in self[0]._fd:
            return self.applyfunc(lambda x: getattr(x, name))
        raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

    def __mul__(self, other):
        if other.is_Function:
            error("Invalid shape, trying to multiply T*a, only a*T supported")
        elif other.is_VectorValued:
            assert other.shape[0] == self.shape[1]
            def entry(i):
                return sum(self[i,k]*other[k] for k in range(self.cols))
            comps = [entry[i] for i in range(self.cols)]
            func = VectorTimeFunction if self.is_TimeDependent or other.is_TimeDependent else VectorFunction
            name = self.name + other.name
            to = getattr(self, 'time_order', 0)
            return func(name=name, grid=self.grid, space_order=self.space_order,
                        components=comps, time_order=to)
        elif other.is_TensorValued:
            assert other.shape[0] == self.shape[1]
            def entry(i, j):
                return sum(self[i,k]*other[k, j] for k in range(self.cols))
            comps = [[entry[i, j] for i in range(self.cols)] for j in range(self.rows)]
            func = TensorTimeFunction if self.is_TimeDependent or other.is_TimeDependent else TensorFunction
            name = self.name + other.name
            to = getattr(self, 'time_order', 0)
            return func(name=name, grid=self.grid, space_order=self.space_order,
                        components=comps, time_order=to)
        else:
            return super(TensorFunction, self).__mul__(other)

    def __rmul__(self, other):
        if other.is_Function:
            return self._eval_scalar_mul(other)
        elif other.is_VectorValued or other.is_TensorValued:
            return (self.T.__mul__(other.T)).T
        else:
            return super(TensorFunction, self).__rmul__(other)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return Function.__dtype_setup__(**kwargs)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return Function.__indices_setup__(**kwargs)

    @property
    def is_diagonal(self):
        return self._is_diagonal

    @property
    def is_symmetric(self):
        return self._is_symmetric

    @property
    def indices(self):
        return self._indices

    @property
    def staggered(self):
        return self._staggered

    @property
    def space_dimensions(self):
        return self.indices

    @property
    def grid(self):
        return self._grid

    @property
    def name(self):
        return self._name

    @property
    def space_order(self):
        return self._space_order

    @property
    def evaluate(self):
        return self.applyfunc(lambda x: x.evaluate)

    def __str__(self):
        name = "SymmetricTensor" if self._is_symmetric else "Tensor"
        if self._is_diagonal:
            name = "DiagonalTensor"
        st = ''.join([' %-2s,' % c for c in self.values()])
        return "%s(%s)"%(name, st)

    __repr__ = __str__

    def _sympy_(self):
        return self
    
    @classmethod
    def _sympify(cls, arg):
        return arg

    def _entry(self, i, j, **kwargs):
        return self.__getitem__(i, j)

    def __getitem__(self, *args):
        if len(args) == 1:
            return super(TensorFunction, self).__getitem__(*args)
        i, j = args
        if self.is_diagonal:
            if i==j:
                return super(TensorFunction, self).__getitem__(i, j)
            return 0.0
        if self.is_symmetric:
            if j < i:
                return super(TensorFunction, self).__getitem__(j, i)
            else:
                return super(TensorFunction, self).__getitem__(i, j)
        return super(TensorFunction, self).__getitem__(i, j)

    @property
    def T(self):
        if self.is_symmetric:
            return self
        else:
            return super(TensorFunction, self).T

    def values(self, symmetric=False):
        if self.is_diagonal:
            return [self[i,i] for i in range(self.shape[0])]
        elif self.is_symmetric:
            val = super(TensorFunction, self).values()
            return list(OrderedDict.fromkeys(val))
        else:
            val = super(TensorFunction, self).values()
            if symmetric:
                shape = self.shape[0]
                inds = np.triu_indices(shape)
                val = val[inds]
            return val
        
        
class TensorTimeFunction(TensorFunction):
    """
    """
    is_TimeDependent = True
    is_TensorValued = True
    _sub_type = TimeFunction
    _time_position = 0

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(TensorTimeFunction, self).__init__(*args, **kwargs)
            self._time_order = kwargs.get('time_order', 1)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return TimeFunction.__indices_setup__(**kwargs)

    @property
    def space_dimensions(self):
        return self.indices[self._time_position+1:]

    @property
    def time_order(self):
        return self._time_order

    @property
    def forward(self):
        """Symbol for the time-forward state of the VectorTimeFunction."""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.indices[self._time_position]

        return self.subs(_t, _t + i * _t.spacing)

    @property
    def backward(self):
        """Symbol for the time-forward state of the VectorTimeFunction."""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.indices[self._time_position]

        return self.subs(_t, _t - i * _t.spacing)


class VectorFunction(TensorFunction):
    """
    """
    is_VectorValued = True
    is_TensorValued = False
    _sub_type = Function
    _time_position = 0

    @property
    def is_symmetric(self):
        return False

    @classmethod
    def __setup_subfunc__(cls, *args, **kwargs):
        comps = kwargs.get("components")
        if comps is not None:
            return comps
        funcs = []
        dims = kwargs.get("grid").dimensions
        stagg = kwargs.get("staggered", dims)
        name = kwargs.get("name")
        for i, d in enumerate(dims):
            kwargs["name"] = name+"_%s"%d.name
            kwargs["staggered"] = d
            funcs.append(cls._sub_type(**kwargs))

        return funcs

    def __str__(self):
        st = ''.join([' %-2s,' % c for c in self])[1:-1]
        return "Vector(%s)"%st

    __repr__ = __str__


class VectorTimeFunction(VectorFunction, TensorTimeFunction):
    """
    """
    is_VectorValued = True
    is_TensorValued = False
    _sub_type = TimeFunction
    _time_position = 0
    is_TimeDependent = True

def sympify_tensor(arg):
    return arg
sympify_converter[TensorFunction] = sympify_tensor
