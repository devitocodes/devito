from collections import OrderedDict

import sympy
import numpy as np
from sympy.core.sympify import converter as sympify_converter
from sympy.core.decorators import call_highest_priority

from devito.finite_differences import Differentiable
from devito.types.basic import AbstractTensor, Basic
from devito.types.dense import Function, TimeFunction
from devito.types.utils import NODE

__all__ = ['TensorFunction', 'TensorTimeFunction', 'VectorFunction', 'VectorTimeFunction']


class TensorFunction(AbstractTensor, Differentiable):
    """
    Tensor valued Function represented as a Matrix.
    Each component is a Function or TimeFunction.

    A TensorFunction and the classes that inherit from it takes the same parameters as
    a DiscreteFunction and additionally:

    Parameters
    ----------
    symmetric : bool, optional
        Whether the tensor is symmetric or not. Defaults to True.
    diagonal : Bool, optional
        Wether the tensor is diagonal or not. Defaults to False.
    staggered: tuple of Dimension, optional
        Staggering of each component, needs to have the size of the tensor. Defaults
        to the Dimensions.
    """
    _sub_type = Function
    _op_priority = Differentiable._op_priority + 2.
    _class_priority = 10

    def __init_finalize__(self, *args, **kwargs):
        self._is_symmetric = kwargs.get('symmetric', True)
        self._is_diagonal = kwargs.get('diagonal', False)
        self._staggered = kwargs.get('staggered', self.space_dimensions)
        self._grid = kwargs.get('grid')
        self._space_order = kwargs.get('space_order', 1)

    @classmethod
    def __subfunc_setup__(cls, *args, **kwargs):
        """
        Creates the components of the TensorFunction
        either from input or from input Dimensions.
        """
        comps = kwargs.get("components")
        if comps is not None:
            return comps
        funcs = []
        grid = kwargs.get("grid")
        if grid is None:
            dims = kwargs.get('dimensions')
            if dims is None:
                raise TypeError("Need either `grid` or `dimensions`")
        else:
            dims = grid.dimensions
        stagg = kwargs.get("staggered", None)
        name = kwargs.get("name")
        symm = kwargs.get('symmetric', True)
        # Fill tensor, only upper diagonal if symmetric
        for i, d in enumerate(dims):
            funcs2 = [0 for _ in range(i)] if symm else []
            start = i if symm else 0
            for j in range(start, len(dims)):
                kwargs["name"] = "%s_%s%s" % (name, d.name, dims[j].name)
                kwargs["staggered"] = (stagg[i][j] if stagg is not None
                                       else (NODE if i == j else (d, dims[j])))
                funcs2.append(cls._sub_type(**kwargs))
            funcs.append(funcs2)

        # Symmetrize and fill diagonal if symmetric
        if symm:
            funcs = np.array(funcs) + np.triu(np.array(funcs), k=1).T
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

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        """
        Multiplication of a TensorFunction T as T*u where u can be a TensorFunction,
        a VectorFunction or a Function/scalar.
        """
        other = sympy.sympify(other)
        # If multiply by a non Devito type defaults to sympy MatVec
        if not isinstance(other, Basic):
            return super(TensorFunction, self).__mul__(other)
        # Scalar Mul
        if other.is_Function:
            return self._eval_scalar_rmul(other)
        # Product of two vector
        elif other.is_VectorValued and self.is_VectorValued:
            # Incorrect size
            if self.is_transposed is other.is_transposed:
                raise ValueError("Incompatible sizes")
            # Inner product
            elif self.is_transposed and not other.is_transposed:
                return sum(s1*s2 for s1, s2 in zip(self, other))
            # Outer product
            else:
                # Only option left but double check
                assert not self.is_transposed and other.is_transposed

                def entry(i, j):
                    return sum(self[i]*other[j] for k in range(self.cols))
                comps = [[entry(i, j) for i in range(self.cols)]
                         for j in range(self.rows)]
                func = tens_func(self, other)
                name = "%s%s" % (self.name, other.name)
                to = getattr(self, 'time_order', 0)
                return func(name=name, grid=self.grid, space_order=self.space_order,
                            components=comps, time_order=to, symmetric=False,
                            diagonal=False)
        # MatVec product
        elif other.is_VectorValued:
            assert other.shape[0] == self.shape[1]

            def entry(i):
                return sum(self[i, k]*other[k] for k in range(self.cols))
            comps = [entry(i) for i in range(self.cols)]
            func = vec_func(self, other)
            name = "%s%s" % (self.name, other.name)
            to = getattr(self, 'time_order', 0)
            return func(name=name, grid=self.grid, space_order=self.space_order,
                        components=comps, time_order=to)
        # MatMat product
        elif other.is_TensorValued:
            assert other.shape[0] == self.shape[1]

            def entry(i, j):
                return sum(self[i, k]*other[k, j] for k in range(self.cols))
            comps = [[entry(i, j) for i in range(self.cols)]
                     for j in range(self.rows)]
            func = tens_func(self, other)
            name = "%s%s" % (self.name, other.name)
            to = getattr(self, 'time_order', 0)
            is_diag = self.is_diagonal and other.is_diagonal
            is_symm = ((self.is_symmetric and other.is_symmetric) or
                       (self.is_symmetric and other.is_diagonal) or
                       (other.is_symmetric and self.is_diagonal))
            return func(name=name, grid=self.grid, space_order=self.space_order,
                        components=comps, time_order=to, symmetric=is_symm,
                        diagonal=is_diag)
        # All cases should be covered but defaults to sympy if in another case
        else:
            return super(TensorFunction, self).__mul__(other)

    def __rmul__(self, other):
        """
        Right multiplication of a TensorFunction T as T*u where u can be a TensorFunction,
        a VectorFunction or a Function/scalar.
        Computes it via mul with tranposes.
        """
        other = sympy.sympify(other)
        try:
            if other.is_Function:
                return self._eval_scalar_mul(other)
            elif other.is_VectorValued or other.is_TensorValued:
                return (self.T.__mul__(other.T)).T
            else:
                return super(TensorFunction, self).__rmul__(other)
        except AttributeError:
            return super(TensorFunction, self).__mul__(other)

    def _eval_at(self, var):
        """
        Evaluate tensor at var location
        """
        def entry(i, j):
            return getattr(self[i, j], '_eval_at', lambda x: self[i, j])(var[i, j])
        comps = [[entry(i, j) for i in range(self.cols)] for j in range(self.rows)]
        to = getattr(self, 'time_order', 0)
        func = tens_func(self, self)
        return func(name='%s%s' % (self.name, var.name),
                    grid=self.grid, space_order=self.space_order,
                    components=comps, time_order=to, symmetric=self.is_symmetric,
                    diagonal=self.is_diagonal)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return Function.__dtype_setup__(**kwargs)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return Function.__indices_setup__(grid=kwargs.get('grid'),
                                          dimensions=kwargs.get('dimensions'))

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
        return self.applyfunc(lambda x: getattr(x, 'evaluate', x))

    # Custom repr and str
    def __str__(self):
        name = "SymmetricTensor" if self._is_symmetric else "Tensor"
        if self._is_diagonal:
            name = "DiagonalTensor"
        st = ''.join([' %-2s,' % c for c in self.values()])
        return "%s(%s)" % (name, st)

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
            if i == j:
                return super(TensorFunction, self).__getitem__(i, j)
            return 0.0
        if self.is_symmetric:
            if j < i:
                return super(TensorFunction, self).__getitem__(j, i)
            else:
                return super(TensorFunction, self).__getitem__(i, j)
        return super(TensorFunction, self).__getitem__(i, j)

    def _eval_transpose(self):
        if self.is_symmetric or self.is_diagonal:
            return self
        mat = [[self[j, i] for j in range(self.cols)] for i in range(self.rows)]
        func = tens_func(self, self)
        name = '%s_T' % self.name
        to = getattr(self, 'time_order', 0)
        return func(name=name, grid=self.grid, space_order=self.space_order,
                    components=mat, time_order=to, symmetric=self.is_symmetric,
                    diagonal=self.is_diagonal)

    def values(self, symmetric=False):
        if self.is_diagonal:
            return [self[i, i] for i in range(self.shape[0])]
        elif self.is_symmetric:
            val = super(TensorFunction, self).values()
            return list(OrderedDict.fromkeys(val))
        else:
            val = super(TensorFunction, self).values()
            if symmetric:
                val = [self[i, j] for j in range(self.cols)
                       for i in range(self.rows) if j >= i]
            return val

    def _eval_add(self, other):
        mat = [[self[i, j] + other[i, j] for j in range(self.cols)]
               for i in range(self.rows)]
        func = tens_func(self, other)
        name = "%s%s" % (self.name, getattr(other, 'name', str(other)))
        to = getattr(self, 'time_order', 0)
        return func(name=name, grid=self.grid, space_order=self.space_order,
                    components=mat, time_order=to, symmetric=self.is_symmetric,
                    diagonal=self.is_diagonal)

    def _eval_scalar_mul(self, other):
        mat = [[self[i, j]*other for j in range(self.cols)] for i in range(self.rows)]
        func = tens_func(self, other)
        name = "%s%s" % (self.name, getattr(other, 'name', str(other)))
        to = getattr(self, 'time_order', 0)
        return func(name=name, grid=self.grid, space_order=self.space_order,
                    components=mat, time_order=to, symmetric=self.is_symmetric,
                    diagonal=self.is_diagonal)

    def _eval_scalar_rmul(self, other):
        return self._eval_scalar_mul(other)

    @property
    def div(self):
        """
        Divergence of the TensorFunction (is a VectorFunction).
        """
        comps = []
        to = getattr(self, 'time_order', 0)
        func = vec_func(self, self)
        for j, d in enumerate(self.space_dimensions):
            comps.append(sum([getattr(self[j, i], 'd%s' % d.name)
                              for i, d in enumerate(self.space_dimensions)]))
        return func(name='div_%s' % self.name, grid=self.grid,
                    space_order=self.space_order, components=comps, time_order=to)

    @property
    def laplace(self):
        """
        Laplacian of the TensorFunction.
        """
        comps = []
        to = getattr(self, 'time_order', 0)
        func = vec_func(self, self)
        for j, d in enumerate(self.space_dimensions):
            comps.append(sum([getattr(self[j, i], 'd%s2' % d.name)
                              for i, d in enumerate(self.space_dimensions)]))
        return func(name='lap_%s' % self.name, grid=self.grid,
                    space_order=self.space_order, components=comps, time_order=to)

    @property
    def grad(self):
        raise AttributeError("Gradient of a second order tensor not supported")

    def new_from_mat(self, mat):
        func = tens_func(self, self)
        name = "%s%s" % ("_", self.name)
        to = getattr(self, 'time_order', 0)
        return func(name=name, grid=self.grid, space_order=self.space_order,
                    components=mat, time_order=to, symmetric=self.is_symmetric,
                    diagonal=self.is_diagonal)


class TensorTimeFunction(TensorFunction):
    """
    Time varying TensorFunction.
    """
    is_TimeDependent = True
    is_TensorValued = True

    _sub_type = TimeFunction
    _time_position = 0

    def __init_finalize__(self, *args, **kwargs):
        super(TensorTimeFunction, self).__init_finalize__(*args, **kwargs)
        self._time_order = kwargs.get('time_order', 1)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return TimeFunction.__indices_setup__(grid=kwargs.get('grid'),
                                              dimensions=kwargs.get('dimensions'))

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
    Vector valued space varying Function as a rank 1 tensor of Function.
    """
    is_VectorValued = True
    is_TensorValued = False

    _sub_type = Function
    _is_symmetric = False

    _op_priority = Differentiable._op_priority + 1.

    def __init_finalize__(self, *args, **kwargs):
        super(VectorFunction, self).__init_finalize__(*args, **kwargs)
        self._is_transposed = kwargs.get("transpose", False)

    @property
    def is_transposed(self):
        return self._is_transposed

    @property
    def shape(self):
        if self.is_transposed:
            return super(VectorFunction, self).shape[::-1]
        else:
            return super(VectorFunction, self).shape

    @classmethod
    def __subfunc_setup__(cls, *args, **kwargs):
        """
        Creates the components of the VectorFunction
        either from input or from input dimensions.
        """
        comps = kwargs.get("components")
        if comps is not None:
            return comps
        funcs = []
        grid = kwargs.get("grid")
        if grid is None:
            dims = kwargs.get('dimensions')
            if dims is None:
                raise TypeError("Need either `grid` or `dimensions`")
        else:
            dims = grid.dimensions
        stagg = kwargs.get("staggered", None)
        name = kwargs.get("name")
        for i, d in enumerate(dims):
            kwargs["name"] = "%s_%s" % (name, d.name)
            kwargs["staggered"] = stagg[i] if stagg is not None else d
            funcs.append(cls._sub_type(**kwargs))

        return funcs

    # Custom repr and str
    def __str__(self):
        st = ''.join([' %-2s,' % c for c in self])[1:-1]
        return "Vector(%s)" % st

    __repr__ = __str__

    # Eval matrices functions for add/mull/evaluate
    def _eval_at(self, var):

        def entry(i):
            return self[i]._eval_at(var[i])

        comps = [entry(i) for i in range(np.max(self.shape))]
        to = getattr(self, 'time_order', 0)
        func = vec_func(self, self)
        return func(name='%s%s' % (self.name, var.name),
                    grid=self.grid, space_order=self.space_order,
                    components=comps, time_order=to)

    def _eval_add(self, other):
        mat = [s+o for s, o in zip(self, other)]
        func = vec_func(self, other)
        name = "%s%s" % (self.name, getattr(other, 'name', str(other)))
        to = getattr(self, 'time_order', 0)
        return func(name=name, grid=self.grid, space_order=self.space_order,
                    components=mat, time_order=to)

    def _eval_scalar_mul(self, other):
        mat = [a*other for a in self._mat]
        func = vec_func(self, other)
        name = "%s%s" % (self.name, getattr(other, 'name', str(other)))
        to = getattr(self, 'time_order', 0)
        return func(name=name, grid=self.grid, space_order=self.space_order,
                    components=mat, time_order=to)

    def _eval_scalar_rmul(self, other):
        return self._eval_scalar_mul(other)

    def _eval_transpose(self):
        func = vec_func(self, self)
        name = '%s_T' % self.name
        to = getattr(self, 'time_order', 0)
        return func(name=name, grid=self.grid, space_order=self.space_order,
                    components=self._mat, time_order=to, transpose=not self.is_transposed)

    @property
    def div(self):
        """
        Divergence of the VectorFunction, creates the divergence Function.
        """
        return sum([getattr(self[i], 'd%s' % d.name)
                    for i, d in enumerate(self.space_dimensions)])

    @property
    def laplace(self):
        """
        Laplacian of the VectorFunction, creates the Laplacian VectorFunction.
        """
        comps = []
        to = getattr(self, 'time_order', 0)
        func = vec_func(self, self)
        comps = [sum([getattr(s, 'd%s2' % d.name) for d in self.space_dimensions])
                 for s in self]
        return func(name='lap_%s' % self.name, grid=self.grid,
                    space_order=self.space_order, components=comps, time_order=to)

    @property
    def curl(self):
        """
        Gradient of the (3D) VectorFunction, creates the curl VectorFunction.
        """

        if len(self.space_dimensions) != 3:
            raise AttributeError("Curl only supported for 3D VectorFunction")
        # The curl of a VectorFunction is a VectorFunction
        derivs = ['d%s' % d.name for d in self.space_dimensions]
        comp1 = getattr(self[2], derivs[1]) - getattr(self[1], derivs[2])
        comp2 = getattr(self[0], derivs[2]) - getattr(self[2], derivs[0])
        comp3 = getattr(self[1], derivs[0]) - getattr(self[0], derivs[1])

        vec_func = VectorTimeFunction if self.is_TimeDependent else VectorFunction
        to = getattr(self, 'time_order', 0)
        return vec_func(name='curl_%s' % self.name, grid=self.grid,
                        space_order=self.space_order, time_order=to,
                        components=[comp1, comp2, comp3])

    @property
    def grad(self):
        """
        Gradient of the VectorFunction, creates the gradient TensorFunction.
        """
        to = getattr(self, 'time_order', 0)
        func = tens_func(self, self)
        comps = [[getattr(f, 'd%s' % d.name) for d in self.space_dimensions]
                 for f in self]
        return func(name='grad_%s' % self.name, grid=self.grid, time_order=to,
                    space_order=self.space_order, components=comps, symmetric=False)

    def new_from_mat(self, mat):
        """
        New VectorFunction with the same property as self and new values
        """
        func = vec_func(self, self)
        name = "%s%s" % ("_", self.name)
        to = getattr(self, 'time_order', 0)
        return func(name=name, grid=self.grid, space_order=self.space_order,
                    components=mat, time_order=to, symmetric=self.is_symmetric,
                    diagonal=self.is_diagonal)


class VectorTimeFunction(VectorFunction, TensorTimeFunction):
    """
    Time varying VectorFunction.
    """
    is_VectorValued = True
    is_TensorValued = False
    is_TimeDependent = True

    _sub_type = TimeFunction
    _time_position = 0

    def func(self):
        return VectorTimeFunction


def vec_func(func1, func2):
    f1 = getattr(func1, 'is_TimeDependent', False)
    f2 = getattr(func2, 'is_TimeDependent', False)
    return VectorTimeFunction if f1 or f2 else VectorFunction


def tens_func(func1, func2):
    f1 = getattr(func1, 'is_TimeDependent', False)
    f2 = getattr(func2, 'is_TimeDependent', False)
    return TensorTimeFunction if f1 or f2 else TensorFunction


def sympify_tensor(arg):
    return arg


sympify_converter[TensorFunction] = sympify_tensor
