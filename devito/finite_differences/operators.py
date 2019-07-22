from devito.logger import error


def div(func):
    """
    Divergence of the input func.
    Func can be a Function, VectorFunction or TensorFunction
    """
    try:
        return func.div
    except AttributeError:
        return 0


def grad(func):
    """
    Gradient of the input func.
    Func can be a Function or a VectorFunction
    """
    try:
        return func.grad
    except AttributeError:
        error("Gradient not supported for class %s" % func.__class__)


def curl(func):
    """
    Curl of the input func.
    Func can be a Function or a VectorFunction
    """
    try:
        return func.curl
    except AttributeError:
        error("Curl only supported for 3D VectorFunction and VectorTimeFunction")


def Laplacian(func):
    """
    Laplacian of the input func.
    Func can be a Function, VectorFunction or TensorFunction
    """
    try:
        return func.laplace
    except AttributeError:
        return 0


def diag(func, size=None):
    """
    Creates the diagonal Tensor with func on its diagonal
    """
    dim = size or len(func.dimensions)
    dim = dim-1 if func.is_TimeDependent else dim
    to = getattr(func, 'time_order', 0)

    from devito.types.tensor import TensorFunction, TensorTimeFunction
    tens_func = TensorTimeFunction if func.is_TimeDependent else TensorFunction

    comps = [[func if i == j else 0 for i in range(dim)] for j in range(dim)]
    return tens_func(name='diag', grid=func.grid, space_order=func.space_order,
                     components=comps, time_order=to, diagonal=True)
