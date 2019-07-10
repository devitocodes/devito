
from devito.finite_differences import Differentiable
from devito.finite_differences.differentiable import Mul, Add
from devito.tools import as_tuple
from devito.types.basic import AbstractCachedSymbol
from devito.types.dense import Function, TimeFunction

__all__ = ['SeparableFunction', 'SeparableTimeFunction']


class SeparableFunction(Differentiable, AbstractCachedSymbol):
    """
    A Function separable in its Dimensions, ie f(x,y,z) = f(x)*f(y)*f(z)

    Take the same parameters as a Function plus

    Parameters
    ----------
    sep: str
        Type of separation as a product or sum of Function, 'sum' or 'prod'.
    separated: Dimension or tuple of Dimensions
        Dimension in which the Function is separable.

    Examples
    --------
    Creation

    >>> from devito import Grid, TimeFunction
    >>> grid = Grid(shape=(4, 4, 5))
    >>> x, y, z = grid.dimensions
    >>> SeparableFunction(name='a', grid=grid)
    a_x(x)*a_y(y)*a_z(z)
    >>> SeparableFunction(name='b', grid=grid, sep='sum')
    b_x(x) + b_y(y) + b_z(z)
    >>> SeparableFunction(name='c', grid=grid, sep='sum', separated=x)
    c_x(x) + c_yz(y, z)
    """
    def __init__(self, *args, **kwargs):
        # Is it sum or product separable
        self._sep_op = spearation_op[kwargs.get('sep', 'prod')]

        # Conventional Function inputs
        self._grid = kwargs.get('grid')
        self._name = kwargs.get('name')
        self._space_order = kwargs.get('space_order', 1)

        # Initialize subfunctions
        # Separable dimensions
        sep = self.separated_dimensions(**kwargs)
        nonseparated = tuple(d for d in self.grid.dimensions if d not in sep)
        dims = tuple(d for d in sep) + (nonseparated,)
        self._dims = dims

        func_list = [self.function(d, **kwargs) for d in sep]
        if len(nonseparated) > 0:
            func_list.append(self.function(nonseparated, **kwargs))

        expr = self._sep_op(*func_list)

        dims = tuple(d for d in sep) + (nonseparated,)
        self._subfunctions = {d: f for d, f in zip(dims, func_list)}
        self._dims = dims
        self._expr = expr
        self._fd = self.expr._fd

    def function(self, dimensions, **kwargs):
        dims = as_tuple(dimensions)
        func_type = Function
        names = "".join(d.name for d in dims)
        shape = tuple(self.grid.dimension_map[d].loc for d in dims)
        kwargs["name"] = '%s_%s' % (self.name, names)
        func = func_type(dimensions=dims, shape=shape, **kwargs)
        return func

    @property
    def sep_op(self):
        return self._sep_op

    @property
    def grid(self):
        return self._grid

    @property
    def space_order(self):
        return self._space_order

    def separated_dimensions(self, **kwargs):
        return as_tuple(kwargs.get('separated', self.grid.dimensions))

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('grid').dtype

    def __repr__(self):
        return self.expr.__repr__()

    __str__ = __repr__

    def func(self, *args, **kwargs):
        return self.from_expr(self.expr.func(*args, **kwargs))

    @property
    def evaluate(self):
        return self.expr.evaluate

    @property
    def expr(self):
        return self._expr

    @property
    def dims(self):
        return self._dims

    @property
    def subfunctions(self):
        return self._subfunctions

    def __getitem__(self, dim):
        return self.subfunctions[dim]

    @property
    def data(self):
        return tuple(f.data for f in self.subfunctions.value)

    def subs(self, *args, **kwargs):
        new = super(SeparableFunction, self).subs(*args, **kwargs)
        new._expr = self.expr.subs(*args, **kwargs)
        return new

    def xreplace(self, rules):
        new = super(SeparableFunction, self).xreplace(rules)
        new._expr = self.expr.xreplace(rules)
        return new


class SeparableTimeFunction(SeparableFunction):
    """
    A TimeFunction separable in its Dimensions, ie f(t,x,y,z) = f(t)*f(x)*f(y)*f(z)

    Take the same parameters as a TimeFunction plus

    Parameters
    ----------
    sep: str
        Type of separation as a product or sum of Function, 'sum' or 'prod'.
    separated: Dimension or tuple of Dimensions
        Dimension in which the TimeFunction is separable.

    Examples
    --------
    Creation

    >>> from devito import Grid, TimeFunction
    >>> grid = Grid(shape=(4, 4, 5))
    >>> x, y, z = grid.dimensions
    >>> t = grid.stepping_dim
    >>> SeparableTimeFunction(name='a', grid=grid, space_order=8, time_order=2)
    a_t(t)*a_x(x)*a_y(y)*a_z(z)
    >>> SeparableTimeFunction(name='b', grid=grid, sep='sum')
    b_t(t) + b_x(x) + b_y(y) + b_z(z)
    >>> SeparableTimeFunction(name='c', grid=grid, sep='sum', separated=t)
    c_t(t) + c_xyz(x, y, z)
    >>> SeparableTimeFunction(name='d', grid=grid, separated=(t, x))
    d_t(t)*d_x(x)*d_yz(y, z)
    """
    def __init__(self, *args, **kwargs):
        self._time_order = kwargs.get('time_order', 1)
        super(SeparableTimeFunction, self).__init__(*args, **kwargs)

    @property
    def time_order(self):
        return self._time_order

    def separated_dimensions(self, **kwargs):
        # Is it sum or product separable
        save = kwargs.get('save')
        time_dim = self.grid.time_dim if isinstance(save, int) else self.grid.stepping_dim
        return as_tuple(kwargs.get('separated', (time_dim,) + self.grid.dimensions))

    def function(self, dimensions, **kwargs):
        dims = as_tuple(dimensions)
        func_type = TimeFunction if any(d.is_Time for d in dims) else Function
        names = "".join(d.name for d in dims)
        save = kwargs.get('save', None)
        shape = tuple(save if d.is_Time else self.grid.dimension_map[d].loc for d in dims)
        kwargs["name"] = '%s_%s' % (self.name, names)
        func = func_type(dimensions=dims, shape=shape, **kwargs)
        return func


spearation_op = {'sum': Add, 'prod': Mul}
