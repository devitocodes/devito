from collections import OrderedDict
from functools import cached_property

import numpy as np
try:
    from sympy.matrices.matrixbase import MatrixBase
except ImportError:
    # Before 1.13
    from sympy.matrices.matrices import MatrixBase
from sympy.core.sympify import converter as sympify_converter

from devito.finite_differences import Differentiable
from devito.finite_differences.tools import make_shift_x0
from devito.types.basic import AbstractTensor
from devito.types.dense import Function, TimeFunction
from devito.types.utils import NODE

__all__ = ['TensorFunction', 'TensorTimeFunction', 'VectorFunction', 'VectorTimeFunction']


class TensorFunction(AbstractTensor):
    """
    Tensor valued Function represented as a Matrix.
    Each component is a Function or TimeFunction.

    A TensorFunction and the classes that inherit from it takes the same parameters as
    a DiscreteFunction and additionally:

    Parameters
    ----------
    name : str
        Name of the symbol.
    grid : Grid, optional
        Carries shape, dimensions, and dtype of the TensorFunction. When grid is not
        provided, shape and dimensions must be given. For MPI execution, a
        Grid is compulsory.
    space_order : int or 3-tuple of ints, optional, default=1
        Discretisation order for space derivatives. ``space_order`` also
        impacts the number of points available around a generic point of interest.  By
        default, ``space_order`` points are available on both sides of a generic point of
        interest, including those nearby the grid boundary. Sometimes, fewer points
        suffice; in other scenarios, more points are necessary. In such cases, instead of
        an integer, one can pass a 3-tuple ``(o, lp, rp)`` indicating the discretization
        order (``o``) as well as the number of points on the left (``lp``) and right
        (``rp``) sides of a generic point of interest.
    shape : tuple of ints, optional
        Shape of the domain region in grid points. Only necessary if ``grid`` isn't given.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if ``grid`` isn't given.
    dtype : data-type, optional, default=np.float32
        Any object that can be interpreted as a numpy data type.
    staggered : Dimension or tuple of Dimension or Stagger, optional
        Staggering of each component, needs to have the size of the tensor. Defaults
        to the Dimensions.
    initializer : callable or any object exposing the buffer interface, optional
        Data initializer. If a callable is provided, data is allocated lazily.
    allocator : MemoryAllocator, optional
        Controller for memory allocation. To be used, for example, when one wants
        to take advantage of the memory hierarchy in a NUMA architecture. Refer to
        `default_allocator.__doc__` for more information.
    padding : int or tuple of ints, optional
        Allocate extra grid points to maximize data access alignment. When a tuple
        of ints, one int per Dimension should be provided.
    symmetric : bool, optional, default=True
        Whether the tensor is symmetric or not.
    diagonal : Bool, optional, default=False
        Whether the tensor is diagonal or not.
    """

    _is_TimeDependent = False

    _sub_type = Function

    _class_priority = 10
    _op_priority = Differentiable._op_priority + 1.

    __rkwargs__ = AbstractTensor.__rkwargs__ + ('dimensions', 'space_order')

    def __init_finalize__(self, *args, **kwargs):
        super().__init_finalize__(*args, **kwargs)
        grid = kwargs.get('grid')
        dimensions = kwargs.get('dimensions')
        inds, _ = Function.__indices_setup__(grid=grid, dimensions=dimensions)
        self._space_dimensions = inds

    @classmethod
    def _component_kwargs(cls, inds, **kwargs):
        """
        Get the kwargs for a single component
        from the kwargs of the TensorFunction.
        """
        kw = {}
        for k, v in kwargs.items():
            if isinstance(v, MatrixBase):
                kw[k] = v[inds]
            else:
                kw[k] = v
        return kw

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
            start = i if (symm or diag) else 0
            stop = i + 1 if diag else len(dims)
            for j in range(start, stop):
                staggj = (stagg[i][j] if stagg is not None
                          else (NODE if i == j else (d, dims[j])))
                sub_kwargs = cls._component_kwargs((i, j), **kwargs)
                sub_kwargs.update({'name': f"{name}_{d.name}{dims[j].name}",
                                   'staggered': staggj})
                funcs2[j] = cls._sub_type(**sub_kwargs)
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
        if name in ['_sympystr', '_pretty', '_latex']:
            return super().__getattr__(self, name)
        try:
            return self.applyfunc(lambda x: x if x == 0 else getattr(x, name))
        except:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

    def _eval_at(self, func):
        """
        Evaluate tensor at func location
        """
        def entries(i, j, func):
            return getattr(self[i, j], '_eval_at', lambda x: self[i, j])(func[i, j])
        entry = lambda i, j: entries(i, j, func)
        return self._new(self.rows, self.cols, entry)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return Function.__indices_setup__(grid=kwargs.get('grid'),
                                          dimensions=kwargs.get('dimensions'))

    @property
    def _symbolic_functions(self):
        return frozenset().union(*[a._symbolic_functions for a in self.values()])

    @property
    def is_TimeDependent(self):
        return self._is_TimeDependent

    @property
    def space_dimensions(self):
        """Spatial dimensions."""
        return self._space_dimensions

    @cached_property
    def root_dimensions(self):
        """Tuple of root Dimensions of the physical space Dimensions."""
        return tuple(d.root for d in self.space_dimensions)

    @cached_property
    def space_order(self):
        """The space order for all components."""
        orders = self.applyfunc(lambda x: x.space_order)
        if len(set(orders)) > 1:
            return orders
        else:
            return orders[0]

    @property
    def is_diagonal(self):
        """Whether the tensor is diagonal."""
        return np.all([self[i, j] == 0 for j in range(self.cols)
                       for i in range(self.rows) if i != j])

    def _evaluate(self, **kwargs):
        def _do_evaluate(x):
            try:
                expand = kwargs.get('expand', True)
                return x._evaluate(expand=expand)
            except AttributeError:
                return x
        return self.applyfunc(_do_evaluate)

    def values(self):
        if self.is_diagonal:
            return [self[i, i] for i in range(self.shape[0])]
        elif self.is_symmetric:
            val = super().values()
            return list(OrderedDict.fromkeys(val))
        else:
            return super().values()

    def div(self, shift=None, order=None, method='FD', **kwargs):
        """
        Divergence of the TensorFunction (is a VectorFunction).

        Parameters
        ----------
        shift: Number, optional, default=None
            Shift for the center point of the derivative in number of gridpoints
        order: int, optional, default=None
            Discretization order for the finite differences.
            Uses `func.space_order` when not specified
        method: str, optional, default='FD'
            Discretization method. Options are 'FD' (default) and
            'RSFD' (rotated staggered grid finite-difference).
        weights/w: list, tuple, or dict, optional, default=None
            Custom weights for the finite differences.
        """
        w = kwargs.get('weights', kwargs.get('w'))
        comps = []
        func = vec_func(self)
        ndim = len(self.space_dimensions)
        space_dims = self.root_dimensions
        shift_x0 = make_shift_x0(shift, (ndim, ndim))
        order = order or self.space_order
        for i in range(len(self.space_dimensions)):
            comps.append(sum([getattr(self[j, i], 'd%s' % d.name)
                              (x0=shift_x0(shift, d, i, j), fd_order=order,
                               method=method, w=w)
                              for j, d in enumerate(space_dims)]))
        return func._new(comps)

    @property
    def laplace(self):
        """
        Laplacian of the TensorFunction.
        """
        return self.laplacian()

    def laplacian(self, shift=None, order=None, method='FD', **kwargs):
        """
        Laplacian of the TensorFunction with shifted derivatives and custom
        FD order.

        Each second derivative is left-right (i.e D^T D with D the first derivative ):
        `(self.dx(x0=dim+shift*dim.spacing,
                  fd_order=order)).dx(x0=dim-shift*dim.spacing, fd_order=order)`

        Parameters
        ----------
        shift: Number, optional, default=None
            Shift for the center point of the derivative in number of gridpoints
        order: int, optional, default=None
            Discretization order for the finite differences.
            Uses `func.space_order` when not specified
        method: str, optional, default='FD'
            Discretization method. Options are 'FD' (default) and
            'RSFD' (rotated staggered grid finite-difference).
        weights/w: list, tuple, or dict, optional, default=None
            Custom weights for the finite
        """
        w = kwargs.get('weights', kwargs.get('w'))
        comps = []
        func = vec_func(self)
        order = order or self.space_order
        space_dims = self.root_dimensions
        ndim = len(self.space_dimensions)
        shift_x0 = make_shift_x0(shift, (ndim, ndim))
        for j in range(ndim):
            comps.append(sum([getattr(self[j, i], 'd%s2' % d.name)
                              (x0=shift_x0(shift, d, j, i), fd_order=order,
                               method=method, w=w)
                              for i, d in enumerate(space_dims)]))
        return func._new(comps)

    def grad(self, shift=None, order=None, method=None, **kwargs):
        raise AttributeError("Gradient of a second order tensor not supported")

    def new_from_mat(self, mat):
        func = tens_func(self)
        return func._new(self.rows, self.cols, mat)

    def classof_prod(self, other, cols):
        is_mat = cols > 1
        is_time = (getattr(self, '_is_TimeDependent', False) or
                   getattr(other, '_is_TimeDependent', False))
        return mat_time_dict[(is_time, is_mat)]


class VectorFunction(TensorFunction):
    """
    Vector valued space varying Function as a rank 1 tensor of Function.
    """

    is_VectorValued = True
    is_TensorValued = False

    _sub_type = Function

    _is_TimeDependent = False

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
            sub_kwargs = cls._component_kwargs(i, **kwargs)
            sub_kwargs.update({'name': f"{name}_{d.name}",
                               'staggered': stagg[i] if stagg is not None else d})
            funcs.append(cls._sub_type(**sub_kwargs))

        return funcs

    # Custom repr and str
    def __str__(self):
        st = ''.join([' %-2s,' % c for c in self])[1:-1]
        return "Vector(%s)" % st

    __repr__ = __str__

    def div(self, shift=None, order=None, method='FD', **kwargs):
        """
        Divergence of the VectorFunction, creates the divergence Function.

        Parameters
        ----------
        shift: Number, optional, default=None
            Shift for the center point of the derivative in number of gridpoints
        order: int, optional, default=None
            Discretization order for the finite differences.
            Uses `func.space_order` when not specified
        method: str, optional, default='FD'
            Discretization method. Options are 'FD' (default) and
            'RSFD' (rotated staggered grid finite-difference).
        weights/w: list, tuple, or dict, optional, default=None
            Custom weights for the finite difference coefficients.
        """
        w = kwargs.get('weights', kwargs.get('w'))
        shift_x0 = make_shift_x0(shift, (len(self.space_dimensions),))
        order = order or self.space_order
        space_dims = self.root_dimensions
        return sum([getattr(self[i], 'd%s' % d.name)(x0=shift_x0(shift, d, None, i),
                                                     fd_order=order, method=method, w=w)
                    for i, d in enumerate(space_dims)])

    @property
    def laplace(self):
        """
        Laplacian of the VectorFunction, creates the Laplacian VectorFunction.
        """
        return self.laplacian()

    def laplacian(self, shift=None, order=None, method='FD', **kwargs):
        """
        Laplacian of the VectorFunction, creates the Laplacian VectorFunction.

        Parameters
        ----------
        shift: Number, optional, default=None
            Shift for the center point of the derivative in number of gridpoints
        order: int, optional, default=None
            Discretization order for the finite differences.
            Uses `func.space_order` when not specified
        method: str, optional, default='FD'
            Discretization method. Options are 'FD' (default) and
            'RSFD' (rotated staggered grid finite-difference).
        weights/w: list, tuple, or dict, optional, default=None
            Custom weights for the finite
        """
        w = kwargs.get('weights', kwargs.get('w'))
        func = vec_func(self)
        shift_x0 = make_shift_x0(shift, (len(self.space_dimensions),))
        order = order or self.space_order
        space_dims = self.root_dimensions
        comps = [sum([getattr(s, 'd%s2' % d.name)(x0=shift_x0(shift, d, None, i),
                                                  fd_order=order, w=w, method=method)
                      for i, d in enumerate(space_dims)])
                 for s in self]
        return func._new(comps)

    def curl(self, shift=None, order=None, method='FD', **kwargs):
        """
        Gradient of the (3D) VectorFunction, creates the curl VectorFunction.

        Parameters
        ----------
        shift: Number, optional, default=None
            Shift for the center point of the derivative in number of gridpoints
        order: int, optional, default=None
            Discretization order for the finite differences.
            Uses `func.space_order` when not specified
        method: str, optional, default='FD'
            Discretization method. Options are 'FD' (default) and
            'RSFD' (rotated staggered grid finite-difference).
        weights/w: list, tuple, or dict, optional, default=None
            Custom weights for the finite difference coefficients.
        """
        if len(self.space_dimensions) != 3:
            raise AttributeError("Curl only supported for 3D VectorFunction")
        # The curl of a VectorFunction is a VectorFunction
        w = kwargs.get('weights', kwargs.get('w'))
        dims = self.root_dimensions
        derivs = ['d%s' % d.name for d in dims]
        shift_x0 = make_shift_x0(shift, (len(dims), len(dims)))
        order = order or self.space_order
        comp1 = (getattr(self[2], derivs[1])(x0=shift_x0(shift, dims[1], 2, 1),
                                             fd_order=order, method=method, w=w) -
                 getattr(self[1], derivs[2])(x0=shift_x0(shift, dims[2], 1, 2),
                                             fd_order=order, method=method, w=w))
        comp2 = (getattr(self[0], derivs[2])(x0=shift_x0(shift, dims[2], 0, 2),
                                             fd_order=order, method=method, w=w) -
                 getattr(self[2], derivs[0])(x0=shift_x0(shift, dims[0], 2, 0),
                                             fd_order=order, method=method, w=w))
        comp3 = (getattr(self[1], derivs[0])(x0=shift_x0(shift, dims[0], 1, 0),
                                             fd_order=order, method=method, w=w) -
                 getattr(self[0], derivs[1])(x0=shift_x0(shift, dims[1], 0, 1),
                                             fd_order=order, method=method, w=w))
        func = vec_func(self)
        return func._new(3, 1, [comp1, comp2, comp3])

    def grad(self, shift=None, order=None, method='FD', **kwargs):
        """
        Gradient of the VectorFunction, creates the gradient TensorFunction.

        Parameters
        ----------
        shift: Number, optional, default=None
            Shift for the center point of the derivative in number of gridpoints
        order: int, optional, default=None
            Discretization order for the finite differences.
            Uses `func.space_order` when not specified
        method: str, optional, default='FD'
            Discretization method. Options are 'FD' (default) and
            'RSFD' (rotated staggered grid finite-difference).
        weights/w: list, tuple, or dict, optional, default=None
            Custom weights for the finite difference coefficients.
        """
        w = kwargs.get('weights', kwargs.get('w'))
        func = tens_func(self)
        ndim = len(self.space_dimensions)
        shift_x0 = make_shift_x0(shift, (ndim, ndim))
        order = order or self.space_order
        space_dims = self.root_dimensions
        comps = [[getattr(f, 'd%s' % d.name)(x0=shift_x0(shift, d, i, j), w=w,
                                             fd_order=order, method=method)
                  for j, d in enumerate(space_dims)]
                 for i, f in enumerate(self)]
        return func._new(comps)

    def outer(self, other):
        comps = [[self[i] * other[j] for i in range(self.cols)] for j in range(self.cols)]
        func = tens_func(self, other)
        return func._new(comps)

    def new_from_mat(self, mat):
        func = vec_func(self)
        return func._new(self.rows, 1, mat)


class TensorTimeFunction(TensorFunction):
    """
    Time varying TensorFunction.
    """

    is_TensorValued = True

    _sub_type = TimeFunction

    _is_TimeDependent = True

    @cached_property
    def time_order(self):
        """The time order for all components."""
        return ({a.time_order for a in self} - {None}).pop()


class VectorTimeFunction(VectorFunction, TensorTimeFunction):
    """
    Time varying VectorFunction.
    """

    is_VectorValued = True
    is_TensorValued = False

    _sub_type = TimeFunction

    _is_TimeDependent = True
    _time_position = 0


mat_time_dict = {(True, True): TensorTimeFunction, (True, False): VectorTimeFunction,
                 (False, True): TensorFunction, (False, False): VectorFunction}


def vec_func(*funcs):
    return (VectorTimeFunction if any([getattr(f, 'is_TimeDependent', False)
            for f in funcs]) else VectorFunction)


def tens_func(*funcs):
    return (TensorTimeFunction if any([getattr(f, 'is_TimeDependent', False)
            for f in funcs]) else TensorFunction)


def sympify_tensor(arg):
    return arg


sympify_converter[TensorFunction] = sympify_tensor
