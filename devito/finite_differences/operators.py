def div(func, shift=None):
    """
    Divergence of the input Function.

    Parameters
    ----------
    func : Function or TensorFunction
    """
    try:
        return func.div(shift=shift)
    except AttributeError:
        return 0


def grad(func, shift=None):
    """
    Gradient of the input Function.

    Parameters
    ----------
    func : Function or VectorFunction
    """
    try:
        return func.grad(shift=shift)
    except AttributeError:
        raise AttributeError("Gradient not supported for class %s" % func.__class__)


def curl(func):
    """
    Curl of the input Function.

    Parameters
    ----------
    func : VectorFunction
    """
    try:
        return func.curl
    except AttributeError:
        raise AttributeError("Curl only supported for 3D VectorFunction")


def laplace(func):
    """
    Laplacian of the input Function.

    Parameters
    ----------
    func : Function or TensorFunction
    """
    try:
        return func.laplace
    except AttributeError:
        return 0


def diag(func, size=None):
    """
    Creates a diagonal tensor with func on its diagonal.

    Parameters
    ----------
    func : Differentiable or scalar
    """
    dim = size or len(func.dimensions)
    dim = dim-1 if func.is_TimeDependent else dim
    to = getattr(func, 'time_order', 0)

    from devito.types.tensor import TensorFunction, TensorTimeFunction
    tens_func = TensorTimeFunction if func.is_TimeDependent else TensorFunction

    comps = [[func if i == j else 0 for i in range(dim)] for j in range(dim)]
    return tens_func(name='diag', grid=func.grid, space_order=func.space_order,
                     components=comps, time_order=to, diagonal=True)
