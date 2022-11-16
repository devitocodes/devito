"""
Machinery to lower np.dtypes and ctypes into strings
"""

import ctypes

import numpy as np
from cgen import Struct, Value, dtype_to_ctype as cgen_dtype_to_ctype

__all__ = ['int2', 'int3', 'int4', 'float2', 'float3', 'float4', 'double2',
           'double3', 'double4', 'dtypes_vector_mapper',
           'dtype_to_cstr', 'dtype_to_ctype', 'dtype_to_mpitype', 'dtype_len',
           'ctypes_to_cstr', 'ctypes_to_cgen', 'c_restrict_void_p',
           'ctypes_vector_mapper', 'is_external_ctype']


# *** Custom np.dtypes

# NOTE: the following is inspired by pyopencl.cltypes

mapper = {
    "int": np.int32,
    "float": np.float32,
    "double": np.float64
}

field_names = ["x", "y", "z", "w"]
counts = [2, 3, 4]
dtypes_vector_mapper = {}
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

        dtypes_vector_mapper[(base_dtype, count)] = dtype


# *** np.dtypes lowering


def dtype_to_cstr(dtype):
    """Translate numpy.dtype into C string."""
    return cgen_dtype_to_ctype(dtype)


def dtype_to_ctype(dtype):
    """Translate numpy.dtype into a ctypes type."""
    try:
        return ctypes_vector_mapper[dtype]
    except KeyError:
        pass
    if issubclass(dtype, ctypes._SimpleCData):
        # Bypass np.ctypeslib's normalization rules such as
        # `np.ctypeslib.as_ctypes_type(ctypes.c_void_p) -> ctypes.c_ulong`
        return dtype
    else:
        return np.ctypeslib.as_ctypes_type(dtype)


def dtype_to_mpitype(dtype):
    """Map numpy types to MPI datatypes."""
    return {np.ubyte: 'MPI_BYTE',
            np.ushort: 'MPI_UNSIGNED_SHORT',
            np.int32: 'MPI_INT',
            int2: 'MPI_INT',
            int3: 'MPI_INT',
            int4: 'MPI_INT',
            np.float32: 'MPI_FLOAT',
            float2: 'MPI_FLOAT',
            float3: 'MPI_FLOAT',
            float4: 'MPI_FLOAT',
            np.int64: 'MPI_LONG',
            np.float64: 'MPI_DOUBLE',
            double2: 'MPI_DOUBLE',
            double3: 'MPI_DOUBLE',
            double4: 'MPI_DOUBLE'}[dtype]


def dtype_len(dtype):
    """
    Number of elements associated with one object of type `dtype`. Thus,
    return 1 for scalar dtypes, N for vector dtypes of rank N.
    """
    try:
        # Vector types
        return len(dtype)
    except TypeError:
        return 1


# *** Custom ctypes


class c_restrict_void_p(ctypes.c_void_p):
    pass


ctypes_vector_mapper = {}
for base_name, base_dtype in mapper.items():
    base_ctype = dtype_to_ctype(base_dtype)

    for count in counts:
        dtype = dtypes_vector_mapper[(base_dtype, count)]

        name = "%s%d" % (base_name, count)
        ctype = type(name, (ctypes.Structure,),
                     {'_fields_': [(i, base_ctype)] for i in field_names[:count]})

        ctypes_vector_mapper[dtype] = ctype


# *** ctypes lowering


def ctypes_to_cstr(ctype, toarray=None, qualifiers=None):
    """Translate ctypes types into C strings."""
    if ctype in ctypes_vector_mapper.values():
        retval = ctype.__name__
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

    # Add qualifiers, if any
    if qualifiers:
        retval = '%s %s' % (' '.join(qualifiers), retval)

    return retval


def ctypes_to_cgen(ctype, fields=None):
    """
    Create a cgen.Structure off a ctypes.Struct, or (possibly nested) ctypes.Pointer
    to ctypes.Struct; return None in all other cases.
    """
    if issubclass(ctype, ctypes._Pointer):
        return ctypes_to_cgen(ctype._type_, fields=fields)
    elif not issubclass(ctype, ctypes.Structure):
        return None

    # Sanity check
    # If a thorough description of the struct fields was required (that is, one
    # symbol per field, each symbol with its own qualifiers), then the number
    # of fields much coincide with that of the `ctype`
    if fields is None:
        fields = (None,)*len(ctype._fields_)
    else:
        assert len(fields) == len(ctype._fields_)

    entries = []
    for i, (n, ct) in zip(fields, ctype._fields_):
        try:
            cstr = i._C_typename

            # Certain qualifiers are to be removed as meaningless within a struct
            degenerate_quals = ('const',)
            cstr = ' '.join([j for j in cstr.split() if j not in degenerate_quals])
        except AttributeError:
            cstr = ctypes_to_cstr(ct)

        # All struct pointers are by construction restrict
        if ct is c_restrict_void_p:
            cstr = '%srestrict' % cstr

        entries.append(Value(cstr, n))

    return Struct(ctype.__name__, entries)


known_ctypes = {
    'vector_types.h': list(ctypes_vector_mapper.values()),
}


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

    for k, v in known_ctypes.items():
        if ctype in v:
            return True

    return False
