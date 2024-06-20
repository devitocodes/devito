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

    if any(f._mem_host for f in functions):
        return False

    fsave = [f for f in functions
             if isinstance(f, TimeFunction) and is_integer(f.save)]

    if 'all-fallback' in gpu_fit and fsave:
        warning("TimeFunction %s assumed to fit the GPU memory" % fsave)
        return True

    return all(f in gpu_fit for f in fsave)


def stream_dimensions(f):
    """
    Return the Function's dimensions that are candidates for data streaming.
    """
    if f.is_TimeFunction:
        return tuple(d for d in f.dimensions if not d.is_Time)
    else:
        return ()


def needs_transfer(f, gpu_fit):
    """
    True if the given object triggers a transfer from/to device memory,
    False otherwise.

    Parameters
    ----------
    f : Function
        The target object.
    gpu_fit : list of Function
        The Function's which are known to definitely fit in the device memory. This
        information is given directly by the user through the compiler option
        `gpu-fit` and is propagated down here through the various stages of lowering.
    """
    return f._mem_mapped and not f.alias and is_on_device(f, gpu_fit)


def is_gpu_create(obj, gpu_create):
    """
    True if the given objects are created and not copied in the device memory,
    False otherwise. Objects created in the device memory are zero-initialised.

    Parameters
    ----------
    obj : Indexed or Function or collection of Functions
        The target object.
    gpu-create : list of Function
        The Function's which are expected to be created in device memory. This
        information is given directly by the user through the compiler option
        `devicecreate` and is propagated down here through the various stages of lowering.
    """
    try:
        functions = (obj.function,)
    except AttributeError:
        functions = as_tuple(obj)

    return all(f in gpu_create for f in functions)


# Import all compiler passes

from .equations import *  # noqa
from .clusters import *  # noqa
from .iet import *  # noqa
