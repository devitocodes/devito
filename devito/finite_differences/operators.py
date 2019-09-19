from devito.logger import error


def div(f):
    """
    Divergence of the input func.

    Parameters
    ----------
    f : Function or TensorFunction
    """
    try:
        return f.div
    except AttributeError:
        return 0


def grad(f):
    """
    Gradient of the input Function.

    Parameters
    ----------
    f : Function or VectorFunction
    """
    try:
        return f.grad
    except AttributeError:
        raise error("Gradient not supported for class %s" % f.__class__)


def curl(f):
    """
    Curl of the input func.

    Parameters
    ----------
    f : VectorFunction
    """
    try:
        return f.curl
    except AttributeError:
        raise AttributeError("Curl only supported for 3D VectorFunction")


def laplace(f):
    """
    Laplacian of the input func.

    Parameters
    ----------
    f : Function or TensorFunction
    """
    try:
        return f.laplace
    except AttributeError:
        return 0


def diag(f, size=None):
    """
    Creates the diagonal tensor with f on its diagonal.

    Parameters
    ----------
    f : Differentiable or scalar
    """
    dim = size or len(f.dimensions)
    dim = dim-1 if f.is_TimeDependent else dim
    to = getattr(f, 'time_order', 0)

    from devito.types.tensor import TensorFunction, TensorTimeFunction
    tens_func = TensorTimeFunction if f.is_TimeDependent else TensorFunction

    comps = [[f if i == j else 0 for i in range(dim)] for j in range(dim)]
    return tens_func(name='diag', grid=f.grid, space_order=f.space_order,
                     components=comps, time_order=to, diagonal=True)
