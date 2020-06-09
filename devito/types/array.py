from ctypes import POINTER
from math import ceil

import numpy as np
from cached_property import cached_property

from devito.parameters import configuration
from devito.tools import ctypes_to_cstr, dtype_to_ctype
from devito.types.basic import AbstractFunction

__all__ = ['Array']


class Array(AbstractFunction):
    """
    Tensor symbol representing an array in symbolic equations.

    An Array is very similar to a sympy.Indexed, though it also carries
    metadata essential for code generation.

    Parameters
    ----------
    name : str
        Name of the symbol.
    dimensions : tuple of Dimension
        Dimensions associated with the object.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    halo : iterable of 2-tuples, optional
        The halo region of the object.
    padding : iterable of 2-tuples, optional
        The padding region of the object.
    scope : str, optional
        Control memory allocation. Allowed values: 'heap', 'stack'. Defaults to 'heap'.
    sharing : str, optional
        Control data sharing. Allowed values: 'shared', 'local'. Defaults to 'shared'.
        'shared' means that in a multi-threaded context, the Array is shared by all
        threads. 'local', instead, means the Array is thread-private.

    Warnings
    --------
    Arrays are created and managed directly by Devito (IOW, they are not
    expected to be used directly in user code).
    """

    is_Array = True
    is_Tensor = True

    def __new__(cls, *args, **kwargs):
        kwargs.update({'options': {'evaluate': False}})
        return AbstractFunction.__new__(cls, *args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super(Array, self).__init_finalize__(*args, **kwargs)

        self._scope = kwargs.get('scope', 'heap')
        assert self._scope in ['heap', 'stack']

        self._sharing = kwargs.get('sharing', 'shared')
        assert self._sharing in ['shared', 'local']

    def __padding_setup__(self, **kwargs):
        padding = kwargs.get('padding')
        if padding is None:
            padding = [(0, 0) for _ in range(self.ndim)]
            if kwargs.get('autopadding', configuration['autopadding']):
                # Heuristic 1; Arrays are typically introduced for temporaries
                # introduced during compilation, and are almost always used together
                # with loop blocking.  Since the typical block size is a multiple of
                # the SIMD vector length, `vl`, padding is made such that the
                # NODOMAIN size is a multiple of `vl` too

                # Heuristic 2: the right-NODOMAIN size is not only a multiple of
                # `vl`, but also guaranteed to be *at least* greater or equal than
                # `vl`, so that the compiler can tweak loop trip counts to maximize
                # the effectiveness of SIMD vectorization

                # Let UB be a function that rounds up a value `x` to the nearest
                # multiple of the SIMD vector length
                vl = configuration['platform'].simd_items_per_reg(self.dtype)
                ub = lambda x: int(ceil(x / vl)) * vl

                fvd_halo_size = sum(self.halo[-1])
                fvd_pad_size = (ub(fvd_halo_size) - fvd_halo_size) + vl

                padding[-1] = (0, fvd_pad_size)
            return tuple(padding)
        elif isinstance(padding, int):
            return tuple((0, padding) for _ in range(self.ndim))
        elif isinstance(padding, tuple) and len(padding) == self.ndim:
            return tuple((0, i) if isinstance(i, int) else i for i in padding)
        else:
            raise TypeError("`padding` must be int or %d-tuple of ints" % self.ndim)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return tuple(kwargs['dimensions']), tuple(kwargs['dimensions'])

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('dtype', np.float32)

    @property
    def shape(self):
        return self.symbolic_shape

    @property
    def scope(self):
        return self._scope

    @property
    def sharing(self):
        return self._sharing

    @property
    def _mem_stack(self):
        return self._scope == 'stack'

    @property
    def _mem_heap(self):
        return self._scope == 'heap'

    @property
    def _mem_local(self):
        return self._sharing == 'local'

    @property
    def _mem_shared(self):
        return self._sharing == 'shared'

    @property
    def _C_typename(self):
        return ctypes_to_cstr(POINTER(dtype_to_ctype(self.dtype)))

    @cached_property
    def free_symbols(self):
        return super().free_symbols - {d for d in self.dimensions if d.is_Default}

    # Pickling support
    _pickle_kwargs = AbstractFunction._pickle_kwargs + ['dimensions', 'scope', 'sharing']
