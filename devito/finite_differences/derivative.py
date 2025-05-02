from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property
from itertools import chain

import sympy

from .finite_difference import generic_derivative, cross_derivative
from .differentiable import Differentiable, diffify, interp_for_fd, Add
from .tools import direct, transpose
from .rsfd import d45
from devito.tools import (as_mapper, as_tuple, frozendict, is_integer,
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
    fd_order : int, tuple of int or dict of {Dimension: int}, optional, default=1
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
        # TODO: Delete this
        if kwargs.get('preprocessed', False):
            from warnings import warn
            warn('I removed the `preprocessed` kwarg')

        if type(expr) is sympy.Derivative:
            raise ValueError("Cannot nest sympy.Derivative with devito.Derivative")
        if not isinstance(expr, Differentiable):
            try:
                expr = diffify(expr)
            except Exception as e:
                raise ValueError("`expr` must be a Differentiable object") from e

        # Validate `dims`. It can be:
        # - a single Dimension ie: x
        # - an iterable of Dimensions ie: (x, y)
        # - a single tuple of Dimension and order ie: (x, 2)
        # - or an iterable of Dimension, order ie: ((x, 2), (y, 2))
        # - any combination of the above ie: ((x, 2), y, x, (z, 3))
        if len(dims) == 0:
            raise ValueError('Expected Dimension w.r.t. which to differentiate')
        elif len(dims) == 1 and isinstance(dims[0], Iterable) and len(dims[0]) != 2:
            # Iterable of Dimensions
            raise ValueError(f'Expected `(dim, deriv_order)`, got {dims[0]}')
        elif len(dims) == 2 and not isinstance(dims[1], Iterable) and is_integer(dims[1]):
            # special case of single dimension and order
            dims = (dims, )

        # Use `deriv_order` if specified
        deriv_order = kwargs.get('deriv_order', (1,)*len(dims))
        if not isinstance(deriv_order, Iterable):
            deriv_order = as_tuple(deriv_order)
        if len(deriv_order) != len(dims):
            raise ValueError(
                'Length of `deriv_order` does not match the length of dimensions'
            )
        if any([not is_integer(d) or d < 0 for d in deriv_order]):
            raise TypeError(
                'Invalid type in `deriv_order`, all elements must be non-negative'
                'Python `int`s'
            )

        # Count the number of derivatives for each dimension
        dcounter = defaultdict(int)
        for d, o in zip(dims, deriv_order):
            if isinstance(d, Iterable):
                if not is_integer(d[1]) or d[1] < 0:
                    raise TypeError(
                        'Invalid type for derivative order, it must be'
                        'non-negative Python `int`'
                    )
                else:
                    dcounter[d[0]] += d[1]
            else:
                dcounter[d] += o

        # Default finite difference orders depending on input dimension (.dt or .dx)
        # It's possible that the expr is a `sympy.Number` at this point, which
        # has derivative 0, unless we're taking a 0th derivative.
        if isinstance(expr, sympy.Number):
            if any(dcounter.values()):
                return 0
            else:
                return expr
        else:
            default_fdo = tuple([
                expr.time_order
                if getattr(d, 'is_Time', False)
                else expr.space_order
                for d in dcounter.keys()
            ])

        # SymPy expects the list of variable w.r.t. which we differentiate to be a list
        # of 2-tuple `(s, count)` where s is the entity to diff wrt and count is the order
        # of the derivative
        derivatives = [sympy.Tuple(d, o) for d, o in dcounter.items()]

        # Construct the actual Derivative object
        obj = Differentiable.__new__(cls, expr, *derivatives)
        obj._dims = tuple(dcounter.keys())

        obj._fd_order = DimensionTuple(
            *as_tuple(kwargs.get('fd_order', default_fdo)),
            getters=obj._dims
        )
        obj._deriv_order = DimensionTuple(
            *as_tuple(dcounter.values()),
            getters=obj._dims
        )
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
        elif isinstance(weights, Differentiable):
            return weights
        else:
            return as_tuple(weights)

    def __call__(self, x0=None, fd_order=None, side=None, method=None, **kwargs):
        weights = kwargs.get('weights', kwargs.get('w'))
        rkw = {}
        if side is not None:
            rkw['side'] = side
        if method is not None:
            rkw['method'] = method
        if weights is not None:
            rkw['weights'] = weights

        if x0 is not None:
            x0 = self._process_x0(self.dims, x0=x0)
            rkw['x0'] = frozendict({**self.x0, **x0})

        if fd_order is not None:
            try:
                _fd_order = dict(fd_order)
            except TypeError:
                assert self.ndims == 1
                _fd_order = {self.dims[0]: fd_order}
            except AttributeError:
                raise TypeError("fd_order incompatible with dimensions")

        if isinstance(self.expr, Derivative):
            # In case this was called on a perfect cross-derivative `u.dxdy`
            # we need to propagate the call to the nested derivative
            rkwe = dict(rkw)
            rkwe.pop('weights', None)
            if 'x0' in rkwe:
                rkwe['x0'] = self._filter_dims(self.expr._filter_dims(rkw['x0']),
                                               neg=True)
            if fd_order is not None:
                fdo = self.expr._filter_dims(_fd_order)
                if fdo:
                    rkwe['fd_order'] = fdo
            rkw['expr'] = self.expr(**rkwe)

        if fd_order is not None:
            rkw['fd_order'] = self._filter_dims(_fd_order, as_tuple=True)

        return self._rebuild(*self.args, **rkw)

    def _rebuild(self, *args, **kwargs):
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
                new, flag = new._xreplace(subs)
                return new, True
            except AttributeError:
                return new, True

        # Resolve nested derivatives
        dsubs = {k: v for k, v in subs.items() if isinstance(k, Derivative)}
        expr = self.expr.xreplace(dsubs)

        subs = self._ppsubs + (subs,)  # Postponed substitutions
        return self._rebuild(expr, *self.args[1:], subs=subs), True

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

        return self._rebuild(*self.args, transpose=adjoint)

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
        if self.expr.staggered == func.staggered and self.expr.is_Function:
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
            return self._rebuild(self.expr._gather_for_diff, *self.args[1:], x0=x0)
        else:
            # For every other cases, that has more functions or more complexe arithmetic,
            # there is not actual way to decide what to do so it’s as safe to use
            # the expression as is.
            return self._rebuild(*self.args, x0=x0)

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
                                   side=self.side, weights=self.weights)
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

    def _eval_expand_nest(self, **hints):
        ''' Expands nested derivatives
        `Derivative(Derivative(f(x), (x, b)), (x, a))
            --> Derivative(f(x), (x, a+b))`
        `Derivative(Derivative(f(x), (y, b)), (x, a))
            --> Derivative(f(x), (x, a), (y, b))`
        Note that this is not always a valid expansion depending on the kwargs
        used to construct the derivative.
        '''
        expr = self.args[0]
        if isinstance(expr, self.__class__):
            new_expr = expr.args[0]
            new_dims = [
                (d, ii)
                for d, ii in zip(
                    chain(self.dims, expr.dims),
                    chain(self.deriv_order, expr.deriv_order)
                )]
            # This is necessary as tools.abc.Reconstructable._rebuild will copy
            # all kwargs from the self object
            # TODO: This dictionary merge needs to be a lot better
            # EG: Don't actually expand if derivatives are incompatible
            new_kwargs = {'deriv_order': tuple(chain(self.deriv_order, expr.deriv_order))}
            return self.func(new_expr, *new_dims, **new_kwargs)
        else:
            return self

    def _eval_expand_mul(self, **hints):
        ''' Expands products, moving independent terms outside the derivative
        `Derivative(C·f(x)·g(c, y), x)
            --> C·g(y)·Derivative(f(x), x)`
        '''
        expr = self.args[0]
        if isinstance(expr, sympy.Mul):
            ind, dep = expr.as_independent(*self.dims, as_Mul=True)
            return ind*self.func(dep, *self.args[1:])
        else:
            return self

    def _eval_expand_add(self, **hints):
        ''' Expands sums, using linearity of derivative
        `Derivative(f(x) + g(x), x)
            --> Derivative(f(x), x) + Derivative(g(x), x)`
        '''
        expr = self.args[0]
        if isinstance(expr, sympy.Add):
            ind, dep = expr.as_independent(*self.dims, as_Add=True)
            if isinstance(dep, sympy.Add):
                return Add(*[self.func(s, *self.args[1:]) for s in dep.args])
            else:
                return self.func(dep, *self.args[1:])
        else:
            return self
