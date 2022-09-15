# The following functions may be used by passes at any layer of abstraction

from devito.logger import warning
from devito.types.dense import TimeFunction
from devito.tools import as_tuple, is_integer


def is_on_device(obj, gpu_fit):
    """
    True if the given object is allocated in the device memory, False otherwise.

    Parameters
    ----------
    obj : Indexed or Function or collection of Functions
        The target object.
    gpu_fit : list of Function
        The Function's which are known to definitely fit in the device memory. This
        information is given directly by the user through the compiler option
        `gpu-fit` and is propagated down here through the various stages of lowering.
    """
    try:
        functions = (obj.function,)
    except AttributeError:
        functions = as_tuple(obj)

    fsave = [f for f in functions
             if isinstance(f, TimeFunction) and is_integer(f.save)]

    if 'all-fallback' in gpu_fit and fsave:
        warning("TimeFunction %s assumed to fit the GPU memory" % fsave)
        return True

    return all(f in gpu_fit for f in fsave)


# Import all compiler passes

from .equations import *  # noqa
from .clusters import *  # noqa
from .iet import *  # noqa
