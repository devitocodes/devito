def div(func, shift=None, order=None, method='FD', **kwargs):
    """
    Divergence of the input Function.

    Parameters
    ----------
    func : Function or TensorFunction
        Function to take the divergence of
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
    try:
        return func.div(shift=shift, order=order, method=method, w=w)
    except AttributeError:
        return 0


def div45(func, shift=None, order=None):
    """
    Divergence of the input Function, using 45 degrees rotated derivative.

    Parameters
    ----------
    func : Function or TensorFunction
        Function to take the divergence of
    shift: Number, optional, default=None
        Shift for the center point of the derivative in number of gridpoints
    order: int, optional, default=None
        Discretization order for the finite differences.
        Uses `func.space_order` when not specified
    """
    return div(func, shift=shift, order=order, method='RSFD')


def grad(func, shift=None, order=None, method='FD', **kwargs):
    """
    Gradient of the input Function.

    Parameters
    ----------
    func : Function or TensorFunction
        Function to take the gradient of
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
    try:
        return func.grad(shift=shift, order=order, method=method, w=w)
    except AttributeError:
        raise AttributeError("Gradient not supported for class %s" % func.__class__)


def grad45(func, shift=None, order=None):
    """
    Gradient of the input Function, using 45 degrees rotated derivative.

    Parameters
    ----------
    func : Function or TensorFunction
        Function to take the gradient of
    shift: Number, optional, default=None
        Shift for the center point of the derivative in number of gridpoints
    order: int, optional, default=None
        Discretization order for the finite differences.
        Uses `func.space_order` when not specified
    """
    return grad(func, shift=shift, order=order, method='RSFD')


def curl(func, shift=None, order=None, method='FD', **kwargs):
    """
    Curl of the input Function. Only supported for VectorFunction

    Parameters
    ----------
    func : VectorFunction
        VectorFunction to take curl of
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
    try:
        return func.curl(shift=shift, order=order, method=method, w=w)
    except AttributeError:
        raise AttributeError("Curl only supported for 3D VectorFunction")


def curl45(func, shift=None, order=None):
    """
    Curl of the input Function, using 45 degrees rotated derivative.
    Only supported for VectorFunction

    Parameters
    ----------
    func : VectorFunction
        VectorFunction to take curl of
    shift: Number, optional, default=None
        Shift for the center point of the derivative in number of gridpoints
    order: int, optional, default=None
        Discretization order for the finite differences.
        Uses `func.space_order` when not specified
    """
    return curl(func, shift=shift, order=order, method='RSFD')


def laplace(func, shift=None, order=None, method='FD', **kwargs):
    """
    Laplacian of the input Function.

    Parameters
    ----------
    func : VectorFunction
        VectorFunction to take laplacian of
    shift: Number, optional, default=None
        Shift for the center point of the derivative in number of gridpoints
    order: int, optional, default=None
        Discretization order for the finite differences.
        Uses `func.space_order` when not specified
    method: str, optional, default='FD'
        Discretization method. Options are 'FD' (default) and 'RSFD'
    weights/w: list, tuple, or dict, optional, default=None
        Custom weights for the finite difference coefficients.
    """
    w = kwargs.get('weights', kwargs.get('w'))
    try:
        return func.laplacian(shift=shift, order=order, method=method, w=w)
    except AttributeError:
        return 0


def diag(func, size=None):
    """
    Creates a diagonal tensor with func on its diagonal.

    Parameters
    ----------
    func : Differentiable or scalar
        Symbolic object to set the diagonal to
    size: int, optional, default=None
        size of the diagonal matrix (size x size).
        Defaults to the number of spatial dimensions when unspecified
    """
    from devito.types.tensor import TensorFunction, TensorTimeFunction
    if isinstance(func, TensorFunction):
        if func.is_TensorValued:
            return func._new(*func.shape, lambda i, j: func[i, i] if i == j else 0)
        else:
            n = func.shape[0]
            return func._new(n, n, lambda i, j: func[i] if i == j else 0)

    dim = size or len(func.dimensions)
    dim = dim-1 if func.is_TimeDependent else dim
    to = getattr(func, 'time_order', 0)

    tens_func = TensorTimeFunction if func.is_TimeDependent else TensorFunction
    comps = [[func if i == j else 0 for i in range(dim)] for j in range(dim)]
    return tens_func(name='diag', grid=func.grid, space_order=func.space_order,
                     components=comps, time_order=to, diagonal=True)
