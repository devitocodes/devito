from collections import OrderedDict

import numpy as np
from sympy.core.sympify import converter as sympify_converter

from devito.finite_differences import DifferentiableMatrix
from devito.types.basic import AbstractTensor
from devito.types.dense import Function, TimeFunction
from devito.types.utils import NODE

__all__ = ['TensorFunction', 'TensorTimeFunction', 'VectorFunction', 'VectorTimeFunction']


class TensorFunction(AbstractTensor, DifferentiableMatrix):
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
        Whether the tensor is diagonal or not. Defaults to False.
    staggered: tuple of Dimension, optional
        Staggering of each component, needs to have the size of the tensor. Defaults
        to the Dimensions.
    """
    _is_TimeDependent = False
    _sub_type = Function
    _op_priority = DifferentiableMatrix._op_priority + 2.
    _class_priority = 10

    def __init_finalize__(self, *args, **kwargs):
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
        diag = kwargs.get('diagonal', False)

        funcs = []
        # Fill tensor, only upper diagonal if symmetric
        for i, d in enumerate(dims):
            funcs2 = [0 for _ in range(len(dims))]
            start = i if symm else 0
            stop = i + 1 if diag else len(dims)
            print(symm, diag, start, stop)
            for j in range(start, stop):
                kwargs["name"] = "%s_%s%s" % (name, d.name, dims[j].name)
                kwargs["staggered"] = (stagg[i][j] if stagg is not None
                                       else (NODE if i == j else (d, dims[j])))
                funcs2[j] = cls._sub_type(**kwargs)
            funcs.append(funcs2)

        # Symmetrize and fill diagonal if symmetric
        if symm:
            funcs = np.array(funcs) + np.triu(np.array(funcs), k=1).T
            funcs = funcs.tolist()
        return funcs, (grid.dim, grid.dim)

    def __getattr__(self, name):
        """
        Try calling a dynamically created FD shortcut.

        Notes
        -----
        This method acts as a fallback for __getattribute__
        """
        try:
            return self.applyfunc(lambda x: getattr(x, name))
        except:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

    def _eval_at(self, func):
        """
        Evaluate tensor at func location
        """
        def entries(i, j, func):
            return getattr(self[i, j], '_eval_at', lambda x: self[i, j])(func[i, j])
        entry = lambda i, j: entries(i, j, func)
        return self.__new__(self.rows, self.cols, entry)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return Function.__dtype_setup__(**kwargs)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return Function.__indices_setup__(grid=kwargs.get('grid'),
                                          dimensions=kwargs.get('dimensions'))

    @property
    def is_TimeDependent(self):
        return self._is_TimeDependent

    @property
    def is_diagonal(self):
        return np.all([self[i, j] == 0  for j in range(self.cols)
                       for i in range(self.rows) if i != j])

    @property
    def is_symmetric(self):
        return np.all(self._comps.T == self._comps)

    @property
    def indices(self):
        return self._indices

    @property
    def dimensions(self):
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
        return "%s(%s)" % (self.name, self.indices)

    __repr__ = __str__

    def _sympy_(self):
        return self

    @classmethod
    def _sympify(cls, arg):
        return arg

    def _entry(self, i, j, **kwargs):
        return self._comps[i, j]

    def values(self):
        if self.is_diagonal:
            return [self[i, i] for i in range(self.shape[0])]
        elif self.is_symmetric:
            val = super(TensorFunction, self).values()
            return list(OrderedDict.fromkeys(val))
        else:
            return super(TensorFunction, self).values()

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
                                              save=kwargs.get('save'),
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

        return self.subs({_t: _t + i * _t.spacing})

    def _eval_scalar_mul(self, other):
        mul = super(TensorFunction, self)._eval_scalar_mul(other)
        if getattr(other, 'is_TimeFunction', False) and not self.is_TimeDependent:
            return mul._as_time(other)
        return mul


class VectorFunction(TensorFunction):
    """
    Vector valued space varying Function as a rank 1 tensor of Function.
    """

    is_VectorValued = True
    is_TensorValued = False
    _is_TimeDependent = False
    _sub_type = Function

    @property
    def is_transposed(self):
        return self.shape[0] == 1

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
            funcs.append([cls._sub_type(**kwargs)])

        return funcs, (grid.dim, 1)

    # Custom repr and str
    def __str__(self):
        st = ''.join([' %-2s,' % c for c in self])[1:-1]
        return "Vector(%s)" % st

    __repr__ = __str__

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

    def _eval_matrix_mul(self, other):
        if self.is_transposed and other.is_TensorValued:
            return (other.T*self.T).T
        return super(VectorFunction, self)._eval_matrix_mul(other)

    def _eval_matrix_rmul(self, other):
        if self.is_transposed and other.is_VectorValued:
            return self.outer(other)
        mul = super(VectorFunction, self)._eval_matrix_rmul(other)
        if not self.is_TimeDependent and other.is_TimeDependent:
            return mul._as_time(other)
        return mul

    def _as_time(self, time_func):
        return VectorTimeFunction(name='%st' % self.name, grid=self.grid,
                                  space_order=self.space_order, components=self._mat,
                                  time_order=time_func.time_order)

    def outer(self, other):
        comps = [[self[i] * other[j] for i in range(self.cols)] for j in range(self.cols)]
        func = tens_func(self, other)
        to = getattr(self, 'time_order', 0)
        return func(name='grad_%s' % self.name, grid=self.grid, time_order=to,
                    space_order=self.space_order, components=comps, symmetric=False)


class TensorTimeFunction(TensorFunction):
    """
    Time varying TensorFunction.
    """
    is_TensorValued = True
    _is_TimeDependent = True
    _sub_type = TimeFunction


class VectorTimeFunction(VectorFunction, TensorTimeFunction):
    """
    Time varying VectorFunction.
    """
    is_VectorValued = True
    is_TensorValued = False
    _is_TimeDependent = True
    _sub_type = TimeFunction
    _time_position = 0


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
