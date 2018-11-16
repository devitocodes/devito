"""The Devito backend configuration engine.

.. warning :: The backend should always be set via :func:`devito.init`
"""

# This code is partly extracted from https://github.com/OP2/PyOP2/

from __future__ import absolute_import

from sympy.core.function import FunctionClass

from devito.exceptions import DevitoError
from devito.logger import warning

backends = {}

backends_registry = ('core', 'yask', 'void', 'ops')


class void(object):
    """
    Dummy backend.
    """
    pass


class _BackendSelector(FunctionClass):

    """
    Metaclass creating the backend class corresponding to the requested class.
    """

    _backend = void

    def __new__(cls, name, bases, dct):
        """
        Inherit Docstrings when creating a class definition. A variation of
        http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
        by Paul McGuire.

        Source: http://stackoverflow.com/a/8101118/396967
        """

        # Get the class docstring
        if not('__doc__' in dct and dct['__doc__']):
            for mro_cls in (cls for base in bases for cls in base.mro()):
                doc = mro_cls.__doc__
                if doc:
                    dct['__doc__'] = doc
                    break
        # Get the attribute docstrings
        for attr, attribute in dct.items():
            if not attribute.__doc__:
                for mro_cls in (cls for base in bases for cls in base.mro()
                                if hasattr(cls, attr)):
                    doc = getattr(getattr(mro_cls, attr), '__doc__')
                    if doc:
                        attribute.__doc__ = doc
                        break
        return type.__new__(cls, name, bases, dct)

    def __call__(cls, *args, **kwargs):
        """
        Create an instance of the request class for the current backend.
        """

        # Try the selected backend first
        try:
            t = cls._backend.__dict__[cls.__name__]
        except KeyError as e:
            warning('Backend %s does not appear to implement class %s'
                    % (cls._backend.__name__, cls.__name__))
            raise e
        # Invoke the constructor with the arguments given
        return t(*args, **kwargs)

    # More disgusting metaclass voodoo
    def __instancecheck__(cls, instance):
        """
        Return True if instance is an instance of cls.

        We need to override the default isinstance check because
        ``type(function.Function(...))`` is ``function.Function``, but
        type(function.Function) is ``_BackendSelector`` and so by default
        ``isinstance(function.Function(...), function.Function)`` is False.
        """
        return isinstance(instance, cls._backend.__dict__[cls.__name__])

    def __subclasscheck__(cls, subclass):
        """
        Return True if subclass is a subclass of cls.

        We need to override the default subclass check because
        ``type(function.Function(...))`` is ``function.Function`, but
        ``type(function.Function)`` is ``_BackendSelector`` and so by default
        ``isinstance(type(function.Function(...)), function.Function)`` is False.
        """
        return issubclass(subclass, cls._backend.__dict__[cls.__name__])


def get_backend():
    """
    Get the Devito backend.
    """
    return _BackendSelector._backend.__name__


def set_backend(backend):
    """
    Set the Devito backend.
    """
    global _BackendSelector
    if _BackendSelector._backend != void:
        warning("WARNING: Switching backend to %s" % backend)

    try:
        # We need to pass a non-empty fromlist so that __import__
        # returns the submodule (i.e. the backend) rather than the
        # package.
        mod = __import__('devito.%s' % backend, fromlist=['None'])
    except ImportError as e:
        warning('Unable to import backend %s' % backend)
        raise e
    backends[backend] = mod
    _BackendSelector._backend = mod


def initialised_backend():
    """Check whether Devito has been yet initialised."""
    return backends.get_backend() != 'void'


def init_backend(backend):
    """
    Initialise Devito: select the backend and other configuration options.
    """
    if backend not in backends_registry:
        raise RuntimeError("Calling init() for a different backend is illegal.")

    try:
        set_backend(backend)
    except (ImportError, RuntimeError):
        raise DevitoError("Couldn't initialize Devito.")
