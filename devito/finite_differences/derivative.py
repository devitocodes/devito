from collections import OrderedDict
from collections.abc import Iterable
from functools import cached_property

import sympy

from .finite_difference import generic_derivative, cross_derivative
from .differentiable import Differentiable, interp_for_fd
from .tools import direct, transpose
from .rsfd import d45
from devito.tools import (as_mapper, as_tuple, filter_ordered, frozendict, is_integer,
                          Pickable)
from devito.types.utils import DimensionTuple

__all__ = ['Derivative']


class Derivative(sympy.Derivative, Differentiable, Pickable):

    """
    An unevaluated Derivative, which carries metadata (Dimensions,
    derivative order, etc) describing how the derivative will be expanded
    upon evaluation.

    Parameters
    ----------
    expr : expr-like
        Expression for which the Derivative is produced.
    dims : Dimension or tuple of Dimension
        Dimensions w.r.t. which to differentiate.
    fd_order : int or tuple of int, optional, default=1
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    deriv_order: int or tuple of int, optional, default=1
        Derivative order.
    side : Side or tuple of Side, optional, default=centered
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1).
    transpose : Transpose, optional, default=direct
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference.
    subs : dict, optional
        Substitutions to apply to the finite-difference expression after evaluation.
    x0 : dict, optional
        Origin (where the finite-difference is evaluated at) for the finite-difference
        scheme, e.g. {x: x, y: y + h_y/2}.

    Examples
    --------
    Creation

    >>> from devito import Function, Derivative, Grid
    >>> grid = Grid((10, 10))
    >>> x, y = grid.dimensions
    >>> u = Function(name="u", grid=grid, space_order=2)
    >>> Derivative(u, x)
    Derivative(u(x, y), x)

    This can also be obtained via the differential shortcut

    >>> u.dx
    Derivative(u(x, y), x)

    You can also specify the order as a keyword argument

    >>> Derivative(u, x, deriv_order=2)
    Derivative(u(x, y), (x, 2))

    Or as a tuple

    >>> Derivative(u, (x, 2))
    Derivative(u(x, y), (x, 2))

    Once again, this can be obtained via shortcut notation

    >>> u.dx2
    Derivative(u(x, y), (x, 2))

    Derivative object are also callable to change default setup:

    >>> u.dx2(x0=x + x.spacing)
    Derivative(u(x, y), (x, 2))

    will create the second derivative at x=x + x.spacing. Accepted arguments for dynamic
    evaluation are `x0`, `fd_order` and `side`.
    """

    _fd_priority = 3

    __rargs__ = ('expr', '*dims')
    __rkwargs__ = ('side', 'deriv_order', 'fd_order', 'transpose', '_ppsubs',
                   'x0', 'method', 'weights')

    def __new__(cls, expr, *dims, **kwargs):
        if type(expr) is sympy.Derivative:
            raise ValueError("Cannot nest sympy.Derivative with devito.Derivative")
        if not isinstance(expr, Differentiable):
            raise ValueError("`expr` must be a Differentiable object")

        new_dims, orders, fd_o, var_count = cls._process_kwargs(expr, *dims, **kwargs)

        # Construct the actual Derivative object
        obj = Differentiable.__new__(cls, expr, *var_count)
        obj._dims = tuple(OrderedDict.fromkeys(new_dims))

        obj._fd_order = DimensionTuple(*as_tuple(fd_o), getters=obj._dims)
        obj._deriv_order = DimensionTuple(*as_tuple(orders), getters=obj._dims)
        obj._side = kwargs.get("side")
        obj._transpose = kwargs.get("transpose", direct)
        obj._method = kwargs.get("method", 'FD')
        obj._weights = cls._process_weights(**kwargs)

        ppsubs = kwargs.get("subs", kwargs.get("_ppsubs", []))
        processed = []
        if ppsubs:
            for i in ppsubs:
                try:
                    processed.append(frozendict(i))
                except AttributeError:
                    # E.g. `i` is a Transform object
                    processed.append(i)
        obj._ppsubs = tuple(processed)

        obj._x0 = cls._process_x0(obj._dims, **kwargs)

        return obj

    @classmethod
    def _process_kwargs(cls, expr, *dims, **kwargs):
        """
        Process arguments for the construction of a Derivative
        """
        # Skip costly processing if constructiong from preprocessed
        if kwargs.get('preprocessed', False):
            fd_orders = kwargs.get('fd_order')
            deriv_orders = kwargs.get('deriv_order')
            if len(dims) == 1:
                dims = tuple([dims[0]]*max(1, deriv_orders[0]))
            variable_count = [sympy.Tuple(s, dims.count(s))
                              for s in filter_ordered(dims)]
            return dims, deriv_orders, fd_orders, variable_count

        # Check `dims`. It can be a single Dimension, an iterable of Dimensions, or even
        # an iterable of 2-tuple (Dimension, deriv_order)
        if len(dims) == 0:
            raise ValueError("Expected Dimension w.r.t. which to differentiate")
        elif len(dims) == 1:
            if isinstance(dims[0], Iterable):
                # Iterable of Dimensions
                if len(dims[0]) != 2:
                    raise ValueError("Expected `(dim, deriv_order)`, got %s" % dims[0])
                orders = kwargs.get('deriv_order', dims[0][1])
                if dims[0][1] != orders:
                    raise ValueError("Two different values of `deriv_order`")
                new_dims = tuple([dims[0][0]]*max(1, dims[0][1]))
            else:
                # Single Dimension
                orders = kwargs.get('deriv_order', 1)
                if isinstance(orders, Iterable):
                    orders = orders[0]
                new_dims = tuple([dims[0]]*max(1, orders))
        elif len(dims) == 2 and not isinstance(dims[1], Iterable) and is_integer(dims[1]):
            # special case of single dimension and order
            orders = dims[1]
            new_dims = tuple([dims[0]]*max(1, orders))
        else:
            # Iterable of 2-tuple, e.g. ((x, 2), (y, 3))
            new_dims = []
            orders = []
            d_ord = kwargs.get('deriv_order', tuple([1]*len(dims)))
            for d, o in zip(dims, d_ord):
                if isinstance(d, Iterable):
                    new_dims.extend([d[0]]*max(1, d[1]))
                    orders.append(d[1])
                else:
                    new_dims.extend([d]*max(1, o))
                    orders.append(o)
            new_dims = as_tuple(new_dims)
            orders = as_tuple(orders)

        # Finite difference orders depending on input dimension (.dt or .dx)
        odims = filter_ordered(new_dims)
        fd_orders = kwargs.get('fd_order', tuple([expr.time_order if
                                                  getattr(d, 'is_Time', False) else
                                                  expr.space_order for d in odims]))
        if len(odims) == 1 and isinstance(fd_orders, Iterable):
            fd_orders = fd_orders[0]

        # SymPy expects the list of variable w.r.t. which we differentiate to be a list
        # of 2-tuple `(s, count)` where s is the entity to diff wrt and count is the order
        # of the derivative
        variable_count = [sympy.Tuple(s, new_dims.count(s))
                          for s in odims]
        return new_dims, orders, fd_orders, variable_count

    @classmethod
    def _process_x0(cls, dims, **kwargs):
        try:
            x0 = frozendict(kwargs.get('x0', {}))
        except TypeError:
            # Only given a value
            _x0 = kwargs.get('x0')
            assert len(dims) == 1 or _x0 is None
            if _x0 is not None and _x0 is not dims[0]:
                x0 = frozendict({dims[0]: _x0})
            else:
                x0 = frozendict({})

        return x0

    @classmethod
    def _process_weights(cls, **kwargs):
        weights = kwargs.get('weights', kwargs.get('w'))
        if weights is None:
            return None
        elif isinstance(weights, sympy.Function):
            return weights
        else:
            return as_tuple(weights)

    def __call__(self, x0=None, fd_order=None, side=None, method=None, weights=None):
        side = side or self._side
        method = method or self._method
        weights = weights if weights is not None else self._weights

        x0 = self._process_x0(self.dims, x0=x0)
        _x0 = frozendict({**self.x0, **x0})

        _fd_order = dict(self.fd_order.getters)
        try:
            _fd_order.update(fd_order or {})
        except TypeError:
            assert self.ndims == 1
            _fd_order.update({self.dims[0]: fd_order or self.fd_order[0]})
        except AttributeError:
            raise TypeError("fd_order incompatible with dimensions")

        if isinstance(self.expr, Derivative):
            # In case this was called on a perfect cross-derivative `u.dxdy`
            # we need to propagate the call to the nested derivative
            x0s = self._filter_dims(self.expr._filter_dims(_x0), neg=True)
            expr = self.expr(x0=x0s, fd_order=self.expr._filter_dims(_fd_order),
                             side=side, method=method)
        else:
            expr = self.expr

        _fd_order = self._filter_dims(_fd_order, as_tuple=True)

        return self._rebuild(fd_order=_fd_order, x0=_x0, side=side, method=method,
                             weights=weights, expr=expr)

    def _rebuild(self, *args, **kwargs):
        kwargs['preprocessed'] = True
        return super()._rebuild(*args, **kwargs)

    func = _rebuild

    def _subs(self, old, new, **hints):
        # Basic case
        if self == old:
            return new
        # Is it in expr?
        if self.expr.has(old):
            newexpr = self.expr._subs(old, new, **hints)
            try:
                return self._rebuild(expr=newexpr)
            except ValueError:
                # Expr replacement leads to non-differentiable expression
                # e.g `f.dx.subs(f: 1) = 1.dx = 0`
                # returning zero
                return sympy.S.Zero

        # In case `x0` was passed as a substitution instead of `(x0=`
        if str(old) == 'x0':
            return self._rebuild(x0={self.dims[0]: new})

        # Trying to substitute by another derivative with different metadata
        # Only need to check if is a Derivative since one for the cases above would
        # have found it
        if isinstance(old, Derivative):
            return self

        # Fall back if we didn't catch any special case
        return self.xreplace({old: new}, **hints)

    def _xreplace(self, subs):
        """
        This is a helper method used internally by SymPy. We exploit it to postpone
        substitutions until evaluation.
        """
        # Return if no subs
        if not subs:
            return self, False

        # Check if trying to replace the whole expression
        if self in subs:
            new = subs.pop(self)
            try:
                return new._xreplace(subs)
            except AttributeError:
                return new, True

        # Resolve nested derivatives
        dsubs = {k: v for k, v in subs.items() if isinstance(k, Derivative)}
        expr = self.expr.xreplace(dsubs)

        subs = self._ppsubs + (subs,)  # Postponed substitutions
        return self._rebuild(subs=subs, expr=expr), True

    @cached_property
    def _metadata(self):
        ret = [self.dims] + [getattr(self, i) for i in self.__rkwargs__]
        ret.append(self.expr.staggered or (None,))
        return tuple(ret)

    def _filter_dims(self, col, as_tuple=False, neg=False):
        """
        Filter collection to only keep the Derivative's dimensions as keys.
        """
        if neg:
            filtered = {k: v for k, v in col.items() if k not in self.dims}
        else:
            filtered = {k: v for k, v in col.items() if k in self.dims}
        if as_tuple:
            return DimensionTuple(*filtered.values(), getters=self.dims)
        else:
            return filtered

    @property
    def dims(self):
        return self._dims

    @property
    def ndims(self):
        return len(self._dims)

    @property
    def x0(self):
        return self._x0

    @property
    def fd_order(self):
        return self._fd_order

    @property
    def deriv_order(self):
        return self._deriv_order

    @property
    def side(self):
        return self._side

    @property
    def transpose(self):
        return self._transpose

    @property
    def is_TimeDependent(self):
        return self.expr.is_TimeDependent

    @property
    def method(self):
        return self._method

    @property
    def weights(self):
        return self._weights

    @property
    def T(self):
        """Transpose of the Derivative.

        FD derivatives can be represented as matrices and have adjoint/transpose.
        This is really useful for more advanced FD definitions. For example
        the conventional Laplacian is `.dxl.T * .dxl`
        """
        if self._transpose == direct:
            adjoint = transpose
        else:
            adjoint = direct

        return self._rebuild(transpose=adjoint)

    def _eval_at(self, func):
        """
        Evaluates the derivative at the location of `func`. It is necessary for staggered
        setup where one could have Eq(u(x + h_x/2), v(x).dx)) in which case v(x).dx
        has to be computed at x=x + h_x/2.
        """
        # If an x0 already exists or evaluating at the same function (i.e u = u.dx)
        # do not overwrite it
        if self.x0 or self.side is not None or func.function is self.expr.function:
            return self
        # For basic equation of the form f = Derivative(g, ...) we can just
        # compare staggering
        if self.expr.staggered == func.staggered:
            return self

        x0 = func.indices_ref.getters
        if self.expr.is_Add:
            # If `expr` has both staggered and non-staggered terms such as
            # `(u(x + h_x/2) + v(x)).dx` then we exploit linearity of FD to split
            # it into `u(x + h_x/2).dx` and `v(x).dx`, since they require
            # different FD indices
            mapper = as_mapper(self.expr._args_diff, lambda i: i.staggered)
            args = [self.expr.func(*v) for v in mapper.values()]
            args.extend([a for a in self.expr.args if a not in self.expr._args_diff])
            args = [self._rebuild(expr=a, x0=x0) for a in args]
            return self.expr.func(*args)
        elif self.expr.is_Mul:
            # For Mul, We treat the basic case `u(x + h_x/2) * v(x) which is what appear
            # in most equation with div(a * u) for example. The expression is re-centered
            # at the highest priority index (see _gather_for_diff) to compute the
            # derivative at x0.
            return self._rebuild(expr=self.expr._gather_for_diff, x0=x0)
        else:
            # For every other cases, that has more functions or more complexe arithmetic,
            # there is not actual way to decide what to do so it’s as safe to use
            # the expression as is.
            return self._rebuild(x0=x0)

    def _evaluate(self, **kwargs):
        # Evaluate finite-difference.
        # NOTE: `evaluate` and `_eval_fd` split for potential future different
        # types of discretizations
        return self._eval_fd(self.expr, **kwargs)

    @property
    def _eval_deriv(self):
        return self._eval_fd(self.expr)

    def _eval_fd(self, expr, **kwargs):
        """
        Evaluate the finite-difference approximation of the Derivative.
        Evaluation is carried out via the following three steps:

        - 1: Interpolate non-derivative shifts.
             E.g u[x, y].dx(x0={y: y + h_y/2}) requires to interpolate `u` in `y`
        - 2: Evaluate derivatives within the expression. For example given
            `f.dx * g`, `f.dx` will be evaluated first.
        - 3: Evaluate the finite difference for the (new) expression.
             This in turn is a two-step procedure, for Functions that may
             may need to be evaluated at a different point due to e.g. a
             shited derivative.
        - 4: Apply substitutions.
        """
        # Step 1: Evaluate non-derivative x0. We currently enforce a simple 2nd order
        # interpolation to avoid very expensive finite differences on top of it
        x0_deriv = self._filter_dims(self.x0)
        x0_interp = {d: v for d, v in self.x0.items()
                     if d not in x0_deriv and not d.is_Time}

        if x0_interp and self.method == 'FD':
            expr = interp_for_fd(expr, x0_interp, **kwargs)

        # Step 2: Evaluate derivatives within expression
        try:
            expr = expr._evaluate(**kwargs)
        except AttributeError:
            pass

        # If True, the derivative will be fully expanded as a sum of products,
        # otherwise an IndexSum will returned
        expand = kwargs.get('expand', True)

        # Step 3: Evaluate FD of the new expression
        if self.method == 'RSFD':
            assert len(self.dims) == 1
            assert self.deriv_order[0] == 1
            res = d45(expr, self.dims[0], x0=self.x0, expand=expand)
        elif len(self.dims) > 1:
            assert self.method == 'FD'
            res = cross_derivative(expr, self.dims, self.fd_order, self.deriv_order,
                                   matvec=self.transpose, x0=x0_deriv, expand=expand,
                                   side=self.side)
        else:
            assert self.method == 'FD'
            res = generic_derivative(expr, self.dims[0], self.fd_order[0],
                                     self.deriv_order[0], weights=self.weights,
                                     side=self.side, matvec=self.transpose,
                                     x0=self.x0, expand=expand)

        # Step 4: Apply substitutions
        for e in self._ppsubs:
            res = res.xreplace(e)

        return res
