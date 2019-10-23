"""
Built-in Operators provided by Devito.
"""

from sympy import Abs, Pow
import numpy as np

import devito as dv
from devito.tools import as_list

__all__ = ['assign', 'smooth', 'gaussian_smooth', 'initialize_function', 'norm',
           'sumall', 'inner', 'mmin', 'mmax']


def assign(f, rhs=0, options=None, name='assign', **kwargs):
    """
    Assign a list of RHSs to a list of Functions.

    Parameters
    ----------
    f : Function or list of Functions
        The left-hand side of the assignment.
    rhs : expr-like or list of expr-like, optional
        The right-hand side of the assignment.
    options : dict or list of dict, optional
        Dictionary or list (of len(f)) of dictionaries containing optional arguments to
        be passed to Eq.
    name : str, optional
        Name of the operator.

    Examples
    --------
    >>> from devito import Grid, Function, assign
    >>> grid = Grid(shape=(4, 4))
    >>> f = Function(name='f', grid=grid, dtype=np.int32)
    >>> g = Function(name='g', grid=grid, dtype=np.int32)
    >>> h = Function(name='h', grid=grid, dtype=np.int32)
    >>> functions = [f, g, h]
    >>> scalars = [1, 2, 3]
    >>> assign(functions, scalars)
    >>> f.data
    Data([[1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1]], dtype=int32)
    >>> g.data
    Data([[2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2]], dtype=int32)
    >>> h.data
    Data([[3, 3, 3, 3],
          [3, 3, 3, 3],
          [3, 3, 3, 3],
          [3, 3, 3, 3]], dtype=int32)
    """
    if not isinstance(rhs, list):
        rhs = len(as_list(f))*[rhs, ]
    eqs = []
    if options:
        for i, j, k in zip(as_list(f), rhs, options):
            if k is not None:
                eqs.append(dv.Eq(i, j, **k))
            else:
                eqs.append(dv.Eq(i, j))
    else:
        for i, j in zip(as_list(f), rhs):
            eqs.append(dv.Eq(i, j))
    dv.Operator(eqs, name=name, **kwargs)()


def smooth(f, g, axis=None):
    """
    Smooth a Function through simple moving average.

    Parameters
    ----------
    f : Function
        The left-hand side of the smoothing kernel, that is the smoothed Function.
    g : Function
        The right-hand side of the smoothing kernel, that is the Function being smoothed.
    axis : Dimension or list of Dimensions, optional
        The Dimension along which the smoothing operation is performed. Defaults
        to ``f``'s innermost Dimension.

    Notes
    -----
    More info about simple moving average available at: ::

        https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    """
    if g.is_Constant:
        # Return a scaled version of the input if it's a Constant
        f.data[:] = .9 * g.data
    else:
        if axis is None:
            axis = g.dimensions[-1]
        dv.Operator(dv.Eq(f, g.avg(dims=axis)), name='smoother')()


def gaussian_smooth(f, sigma=1, _order=4, mode='reflect'):
    """
    Gaussian smooth function.
    """
    class ObjectiveDomain(dv.SubDomain):

        name = 'objective_domain'

        def __init__(self, lw):
            super(ObjectiveDomain, self).__init__()
            self.lw = lw

        def define(self, dimensions):
            return {d: ('middle', self.lw, self.lw) for d in dimensions}

    def fset(f, g):
        indices = [slice(lw, -lw, 1) for _ in g.grid.dimensions]
        slices = (slice(None, None, 1), )*len(g.grid.dimensions)
        if isinstance(f, np.ndarray):
            f[slices] = g.data[tuple(indices)]
        elif isinstance(f, dv.Function):
            f.data[slices] = g.data[tuple(indices)]
        else:
            raise NotImplementedError

    lw = int(_order*sigma + 0.5)

    # Create the padded grid:
    objective_domain = ObjectiveDomain(lw)
    try:
        shape_padded = np.array(f.grid.shape) + 2*lw
    except AttributeError:
        shape_padded = np.array(f.shape) + 2*lw
    grid = dv.Grid(shape=shape_padded, subdomains=objective_domain)

    f_c = dv.Function(name='f_c', grid=grid, space_order=2*lw,
                      coefficients='symbolic', dtype=np.int32)
    f_o = dv.Function(name='f_o', grid=grid, coefficients='symbolic', dtype=np.int32)

    weights = np.exp(-0.5/sigma**2*(np.linspace(-lw, lw, 2*lw+1))**2)
    weights = weights/weights.sum()

    mapper = {}
    for d in f_c.dimensions:
        lhs = []
        rhs = []
        options = []

        lhs.append(f_o)
        rhs.append(dv.generic_derivative(f_c, d, 2*lw, 1))
        coeffs = dv.Coefficient(1, f_c, d, weights)
        options.append({'coefficients': dv.Substitutions(coeffs),
                        'subdomain': grid.subdomains['objective_domain']})
        lhs.append(f_c)
        rhs.append(f_o)
        options.append({'subdomain': grid.subdomains['objective_domain']})

        mapper[d] = {'lhs': lhs, 'rhs': rhs, 'options': options}

    initialize_function(f_c, f.data[:], lw,
                        mapper=mapper,
                        mode='reflect', name='smooth')

    fset(f, f_c)

    return f


def initialize_function(function, data, nbl, mapper=None, mode='constant',
                        name='padfunc'):
    """
    Initialize a Function with the given ``data``. ``data``
    does *not* include the ``nbl`` outer/boundary layers; these are added via padding
    by this function.

    Parameters
    ----------
    function : Function
        The initialised object.
    data : ndarray or Function
        The data used for initialisation.
    nbl : int
        Number of outer layers (such as absorbing layers for boundary damping).
    mapper : dict, optional
        Dictionary containing, for each dimension of `function`, a sub-dictionary
        containing the following keys:
        1) 'lhs': List of additional expressions to be added to the LHS expressions list.
        2) 'rhs': List of additional expressions to be added to the RHS expressions list.
        3) 'options': Options pertaining to the additional equations that will be
        constructed.
    mode : str, optional
        The function initialisation mode. 'constant' and 'reflect' are
        accepted.
    name : str, optional
        The name assigned to the operator.

    Examples
    --------
    In the following example the `'interior'` of a function is set to one plus
    the value on the boundary.

    >>> import numpy as np
    >>> from devito import Grid, SubDomain, Function, initialize_function

    Create the computational domain:

    >>> grid = Grid(shape=(6, 6))
    >>> x, y = grid.dimensions

    Create the Function we wish to set along with the data to set it:

    >>> f = Function(name='f', grid=grid, dtype=np.int32)
    >>> data = np.full((4, 4), 2, dtype=np.int32)

    Now create the additional expressions and options required to set the value of
    the interior region to one greater than the boundary value. Note that the equation
    is specified on the second (final) grid dimension so that additional equation is
    executed after padding is complete.

    >>> lhs = f
    >>> rhs = f+1
    >>> options = {'subdomain': grid.subdomains['interior']}
    >>> mapper = {}
    >>> mapper[y] = {'lhs': lhs, 'rhs': rhs, 'options': options}

    Call the initialize_function routine:

    >>> initialize_function(f, data, 1, mapper=mapper)
    >>> f.data
    Data([[2, 2, 2, 2, 2, 2],
          [2, 3, 3, 3, 3, 2],
          [2, 3, 3, 3, 3, 2],
          [2, 3, 3, 3, 3, 2],
          [2, 3, 3, 3, 3, 2],
          [2, 2, 2, 2, 2, 2]], dtype=int32)
    """
    if isinstance(function, dv.TimeFunction):
        raise NotImplementedError("TimeFunctions are not currently supported.")

    slices = tuple([slice(nbl, -nbl) for _ in range(function.grid.dim)])
    if isinstance(data, dv.Function):
        function.data[slices] = data.data[:]
    else:
        function.data[slices] = data
    lhs = []
    rhs = []
    options = []

    if mode == 'reflect' and function.grid.distributor.is_parallel:
        # Check that HALO size is appropriate
        halo = function.halo
        local_size = function.shape

        def buff(i, j):
            return [(i + k - 2*nbl) for k in j]

        b = [min(l) for l in (w for w in (buff(i, j) for i, j in zip(local_size, halo)))]
        if any(np.array(b) < 0):
            raise ValueError("Function `%s` halo is not sufficiently thick." % function)

    for d in function.space_dimensions:
        dim_l = dv.SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                     thickness=nbl)
        dim_r = dv.SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                      thickness=nbl)
        if mode == 'constant':
            subsl = nbl
            subsr = d.symbolic_max - nbl
        elif mode == 'reflect':
            subsl = 2*nbl - 1 - dim_l
            subsr = 2*(d.symbolic_max - nbl) + 1 - dim_r
        else:
            raise ValueError("Mode not available")
        lhs.append(function.subs({d: dim_l}))
        lhs.append(function.subs({d: dim_r}))
        rhs.append(function.subs({d: subsl}))
        rhs.append(function.subs({d: subsr}))
        options.extend([None, None])

        if mapper and d in mapper.keys():
            exprs = mapper[d]
            lhs_extra = exprs['lhs']
            rhs_extra = exprs['rhs']
            lhs.extend(as_list(lhs_extra))
            rhs.extend(as_list(rhs_extra))
            options_extra = exprs.get('options', len(as_list(lhs_extra))*[None, ])
            if isinstance(options_extra, list):
                options.extend(options_extra)
            else:
                options.extend([options_extra])

    if all(options is None for i in options):
        options = None

    # TODO: Figure out why yask doesn't like it with dse/dle
    assign(lhs, rhs, options=options, name=name, dse='noop', dle='noop')


# Reduction-inducing builtins

class MPIReduction(object):
    """
    A context manager to build MPI-aware reduction Operators.
    """

    def __init__(self, *functions, op=dv.mpi.MPI.SUM):
        grids = {f.grid for f in functions}
        if len(grids) == 0:
            self.grid = None
        elif len(grids) == 1:
            self.grid = grids.pop()
        else:
            raise ValueError("Multiple Grids found")
        dtype = {f.dtype for f in functions}
        if len(dtype) == 1:
            self.dtype = dtype.pop()
        else:
            raise ValueError("Illegal mixed data types")
        self.v = None
        self.op = op

    def __enter__(self):
        i = dv.Dimension(name='i',)
        self.n = dv.Function(name='n', shape=(1,), dimensions=(i,),
                             grid=self.grid, dtype=self.dtype)
        self.n.data[0] = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.grid is None or not dv.configuration['mpi']:
            assert self.n.data.size == 1
            self.v = self.n.data[0]
        else:
            comm = self.grid.distributor.comm
            self.v = comm.allreduce(np.asarray(self.n.data), self.op)[0]


@dv.switchconfig(log_level='ERROR')
def norm(f, order=2):
    """
    Compute the norm of a Function.

    Parameters
    ----------
    f : Function
        Input Function.
    order : int, optional
        The order of the norm. Defaults to 2.
    """
    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    with MPIReduction(f) as mr:
        op = dv.Operator(eqns + [dv.Inc(mr.n[0], Abs(Pow(p, order)))],
                         name='norm%d' % order)
        op.apply(**kwargs)

    v = Pow(mr.v, 1/order)

    return np.float(v)


def sumall(f):
    """
    Compute the sum of all Function data.

    Parameters
    ----------
    f : Function
        Input Function.
    """
    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    p, eqns = f.guard() if f.is_SparseFunction else (f, [])

    with MPIReduction(f) as mr:
        op = dv.Operator(eqns + [dv.Inc(mr.n[0], p)], name='sum')
        op.apply(**kwargs)

    return np.float(mr.v)


def inner(f, g):
    """
    Inner product of two Functions.

    Parameters
    ----------
    f : Function
        First input operand
    g : Function
        Second input operand

    Raises
    ------
    ValueError
        If the two input Functions are defined over different grids, or have
        different dimensionality, or their dimension-wise sizes don't match.
        If in input are two SparseFunctions and their coordinates don't match,
        the exception is raised.

    Notes
    -----
    The inner product is the sum of all dimension-wise products. For 1D Functions,
    the inner product corresponds to the dot product.
    """
    # Input check
    if f.is_TimeFunction and f._time_buffering != g._time_buffering:
        raise ValueError("Cannot compute `inner` between save/nosave TimeFunctions")
    if f.shape != g.shape:
        raise ValueError("`f` and `g` must have same shape")
    if f._data is None or g._data is None:
        raise ValueError("Uninitialized input")
    if f.is_SparseFunction and not np.all(f.coordinates_data == g.coordinates_data):
        raise ValueError("Non-matching coordinates")

    kwargs = {}
    if f.is_TimeFunction and f._time_buffering:
        kwargs[f.time_dim.max_name] = f._time_size - 1

    # Protect SparseFunctions from accessing duplicated (out-of-domain) data,
    # otherwise we would eventually be summing more than expected
    rhs, eqns = f.guard(f*g) if f.is_SparseFunction else (f*g, [])

    with MPIReduction(f, g) as mr:
        op = dv.Operator(eqns + [dv.Inc(mr.n[0], rhs)], name='inner')
        op.apply(**kwargs)

    return np.float(mr.v)


def mmin(f):
    """
    Retrieve the minimum.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """
    if isinstance(f, dv.Constant):
        return f.data
    elif isinstance(f, dv.Function):
        with MPIReduction(f, op=dv.mpi.MPI.MIN) as mr:
            mr.n.data[0] = np.min(f.data_ro_domain).item()
        return mr.v.item()
    else:
        raise ValueError("Expected Function, not `%s`" % type(f))


def mmax(f):
    """
    Retrieve the maximum.

    Parameters
    ----------
    f : array_like or Function
        Input operand.
    """
    if isinstance(f, dv.Constant):
        return f.data
    elif isinstance(f, dv.Function):
        with MPIReduction(f, op=dv.mpi.MPI.MAX) as mr:
            mr.n.data[0] = np.max(f.data_ro_domain).item()
        return mr.v.item()
    else:
        raise ValueError("Expected Function, not `%s`" % type(f))
