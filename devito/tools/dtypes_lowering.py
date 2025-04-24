"""
Machinery to lower np.dtypes and ctypes into strings
"""

import ctypes
from functools import reduce
from operator import mul

import numpy as np
from cgen import dtype_to_ctype as cgen_dtype_to_ctype

from .utils import as_tuple

__all__ = ['int2', 'int3', 'int4', 'float2', 'float3', 'float4', 'double2',  # noqa
           'double3', 'double4', 'dtypes_vector_mapper', 'dtype_to_mpidtype',
           'dtype_to_cstr', 'dtype_to_ctype', 'infer_datasize', 'dtype_to_mpitype',
           'dtype_len', 'ctypes_to_cstr', 'c_restrict_void_p', 'ctypes_vector_mapper',
           'is_external_ctype', 'infer_dtype', 'extract_dtype', 'CustomDtype',
           'mpi4py_mapper']


# *** Custom np.dtypes

# NOTE: the following is inspired by pyopencl.cltypes

dtype_mapper = {
    "int": np.int32,
    "float": np.float32,
    "double": np.float64
}


def build_dtypes_vector(field_names, counts, mapper=None):
    ret = {}
    mapper = mapper or dtype_mapper
    for base_name, base_dtype in mapper.items():
        for count in counts:
            name = "%s%d" % (base_name, count)

            titles = field_names[:count]

            padded_count = count
            if count == 3:
                padded_count = 4

            names = ["s%d" % i for i in range(count)]
            while len(names) < padded_count:
                names.append("padding%d" % (len(names) - count))

            if len(titles) < len(names):
                titles.extend((len(names) - len(titles)) * [None])

            dtype = np.dtype(dict(
                names=names,
                formats=[base_dtype] * padded_count,
                titles=titles))

            globals()[name] = dtype

            ret[(base_dtype, count)] = dtype

    return ret


field_names = ["x", "y", "z", "w"]
counts = [2, 3, 4]


class DTypesVectorMapper(dict):

    def add_dtype(self, field_name, count):
        """
        Build and return a vector dtype with `count` > 4 fields, that is a
        non-canonical vector dtype as normally encountered in e.g. float3 in
        CUDA or HIP.
        """
        if not isinstance(field_name, str) or field_name in field_names:
            raise ValueError("Expected field name different than x,y,z,w")
        if count <= max(counts):
            raise ValueError("Expected at least 5 fields")

        self.update(build_dtypes_vector([field_name], [count]))

    def get_base_dtype(self, v, default=None):
        for (base_dtype, count), dtype in self.items():
            if dtype is v:
                return base_dtype

        if default is not None:
            return default
        else:
            raise ValueError


dtypes_vector_mapper = DTypesVectorMapper()
# Standard vector dtypes
dtypes_vector_mapper.update(build_dtypes_vector(field_names, counts))
# Fallbacks
dtypes_vector_mapper.update({(v, 1): v for v in dtype_mapper.values()})


# *** Custom types escaping both the numpy and ctypes namespaces


class CustomDtype:

    def __init__(self, name, template=None, modifier=None):
        self.name = name
        self.template = as_tuple(template)
        self.modifier = modifier or ''

    def __eq__(self, other):
        return (isinstance(other, CustomDtype) and
                self.name == other.name and
                self.template == other.template and
                self.modifier == other.modifier)

    def __hash__(self):
        return hash((self.name, self.template, self.modifier))

    def __repr__(self):
        template = '<%s>' % ','.join([str(i) for i in self.template])
        return "%s%s%s" % (self.name,
                           template if self.template else '',
                           self.modifier)

    __str__ = __repr__


# *** np.dtypes lowering


def dtype_to_cstr(dtype):
    """Translate numpy.dtype into C string."""
    return cgen_dtype_to_ctype(dtype)


def dtype_to_ctype(dtype):
    """Translate numpy.dtype into a ctypes type."""
    if isinstance(dtype, CustomDtype):
        return dtype

    try:
        return ctypes_vector_mapper[dtype]
    except KeyError:
        pass

    if isinstance(dtype, CustomDtype):
        return dtype
    elif issubclass(dtype, (ctypes._Pointer, ctypes.Structure, ctypes._SimpleCData)):
        # Bypass np.ctypeslib's normalization rules such as
        # `np.ctypeslib.as_ctypes_type(ctypes.c_void_p) -> ctypes.c_ulong`
        return dtype
    else:
        return np.ctypeslib.as_ctypes_type(dtype)


def infer_datasize(dtype, shape):
    """
    Translate numpy.dtype to (ctype, int): type and scale for correct C allocation size.
    """
    datasize = int(reduce(mul, shape))
    if isinstance(dtype, CustomDtype):
        return dtype, datasize

    try:
        return ctypes_vector_mapper[dtype], datasize
    except KeyError:
        pass

    if issubclass(dtype, ctypes._SimpleCData):
        return dtype, datasize

    if dtype == np.float16:
        # Allocate half float as unsigned short
        return ctypes.c_uint16, datasize

    if np.issubdtype(dtype, np.complexfloating):
        # For complex float, allocate twice the size of real/imaginary part
        return np.ctypeslib.as_ctypes_type(dtype(0).real.__class__), 2 * datasize

    return np.ctypeslib.as_ctypes_type(dtype), datasize


mpi4py_mapper = {}
mpi_mapper = {
    np.ubyte: 'MPI_BYTE',
    np.ushort: 'MPI_UNSIGNED_SHORT',
    np.int32: 'MPI_INT',
    np.float32: 'MPI_FLOAT',
    np.int64: 'MPI_LONG',
    np.float64: 'MPI_DOUBLE',
    np.complex64: 'MPI_C_COMPLEX',
    np.complex128: 'MPI_C_DOUBLE_COMPLEX'
}


def dtype_to_mpitype(dtype):
    """Map numpy types to MPI datatypes."""

    # Resolve vector dtype if necessary
    dtype = dtypes_vector_mapper.get_base_dtype(dtype)

    return mpi_mapper[dtype]


def dtype_to_mpidtype(dtype):
    """
    Map numpy type to MPI internal types for communication
    """
    from devito.mpi import MPI
    return MPI._typedict[np.dtype(dtype).char]


def dtype_len(dtype):
    """
    Number of elements associated with one object of type `dtype`. Thus,
    return 1 for scalar dtypes, N for vector dtypes of rank N.
    """
    try:
        # Vector types
        dlen = len(dtype)
        if 'padding0' in dtype.fields:
            dlen -= 1
        return dlen
    except TypeError:
        return 1


# *** Custom ctypes


class c_restrict_void_p(ctypes.c_void_p):
    pass


ctypes_vector_mapper = {}
for base_name, base_dtype in dtype_mapper.items():
    base_ctype = dtype_to_ctype(base_dtype)

    for count in counts:
        dtype = dtypes_vector_mapper[(base_dtype, count)]

        name = "%s%d" % (base_name, count)
        ctype = type(name, (ctypes.Structure,),
                     {'_fields_': [(i, base_ctype) for i in field_names[:count]],
                      '_base_dtype': True})

        ctypes_vector_mapper[dtype] = ctype


# *** ctypes lowering


def ctypes_to_cstr(ctype, toarray=None):
    """Translate ctypes types into C strings."""
    if isinstance(ctype, str):
        # Already a C string
        return ctype
    elif ctype in ctypes_vector_mapper.values():
        retval = ctype.__name__
    elif isinstance(ctype, CustomDtype):
        retval = str(ctype)
    elif issubclass(ctype, ctypes.Structure):
        retval = 'struct %s' % ctype.__name__
    elif issubclass(ctype, ctypes.Union):
        retval = 'union %s' % ctype.__name__
    elif issubclass(ctype, ctypes._Pointer):
        if toarray:
            retval = ctypes_to_cstr(ctype._type_, '(* %s)' % toarray)
        else:
            retval = ctypes_to_cstr(ctype._type_)
            if issubclass(ctype._type_, ctypes._Pointer):
                # Look-ahead to avoid extra ugly spaces
                retval = '%s*' % retval
            else:
                retval = '%s *' % retval
    elif issubclass(ctype, ctypes.Array):
        retval = '%s[%d]' % (ctypes_to_cstr(ctype._type_, toarray), ctype._length_)
    elif ctype.__name__.startswith('c_'):
        name = ctype.__name__[2:]
        # A primitive datatype
        # FIXME: Is there a better way of extracting the C typename ?
        # Here, we're following the ctypes convention that each basic type has
        # either the format c_X_p or c_X, where X is the C typename, for instance
        # `int` or `float`.
        if name.endswith('_p'):
            name = name.split('_')[-2]
            suffix = '*'
        elif toarray:
            suffix = toarray
        else:
            suffix = None

        if name.startswith('u'):
            name = name[1:]
            prefix = 'unsigned'
        else:
            prefix = None

        # Special cases
        if name == 'byte':
            name = 'char'

        retval = name
        if prefix:
            retval = '%s %s' % (prefix, retval)
        if suffix:
            retval = '%s %s' % (retval, suffix)
    else:
        # A custom datatype (e.g., a typedef-ed pointer to struct)
        retval = ctype.__name__

    return retval


def is_external_ctype(ctype, includes):
    """
    True if `ctype` is known to be declared in one of the given `includes`
    files, False otherwise.
    """
    # Get the base type
    while issubclass(ctype, ctypes._Pointer):
        ctype = ctype._type_

    if issubclass(ctype, ctypes._SimpleCData):
        return False

    if ctype in ctypes_vector_mapper.values():
        return True

    return False


def is_numpy_dtype(dtype):
    """
    True if `dtype` is a numpy dtype, False otherwise.
    """
    try:
        return issubclass(dtype, np.generic)
    except TypeError:
        return False


def infer_dtype(dtypes):
    """
    Given a set of np.dtypes, return the "winning" dtype:

        * In the case of multiple floating dtypes, return the dtype with
          highest precision;
        * If there's at least one floating dtype, ignore any integer dtypes.
    """
    # Resolve the vector types, if any
    dtypes = {dtypes_vector_mapper.get_base_dtype(i, i) for i in dtypes}
    # Only keep number types
    dtypes = {i for i in dtypes if is_numpy_dtype(i)}
    # Separate floating point types from the rest
    fdtypes = {i for i in dtypes if np.issubdtype(i, np.floating) or
               np.issubdtype(i, np.complexfloating)}
    if len(fdtypes) > 1:
        return max(fdtypes, key=lambda i: np.dtype(i).itemsize)
    elif len(fdtypes) == 1:
        return fdtypes.pop()
    elif len(dtypes) == 1:
        return dtypes.pop()
    else:
        # E.g., mixed integer arithmetic
        return max(dtypes, key=lambda i: np.dtype(i).itemsize, default=None)


def extract_dtype(expr):
    """Extract the "winning" dtype from an expression"""
    dtypes = {getattr(e, 'dtype', None)
              for e in expr.free_symbols}
    return infer_dtype(dtypes - {None})
