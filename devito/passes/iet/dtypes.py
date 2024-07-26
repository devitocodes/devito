import numpy as np
import ctypes

from devito.ir import FindSymbols
from devito.ir.iet.nodes import Dereference
from devito.ir.iet.visitors import Uxreplace
from devito.symbolics.extended_dtypes import Float16P
from devito.tools.utils import as_list
from devito.types.basic import Symbol

__all__ = ['lower_dtypes']


def lower_dtypes(iet, lang, compiler, sregistry):
    """
    Lowers float16 scalar types to pointers since we can't directly pass their
    value. Also includes headers for complex arithmetic if needed.
    """

    iet, metadata = _complex_includes(iet, lang, compiler)

    # Lower float16 parameters to pointers and dereference
    body_prefix = []
    body_mapper = {}
    params_mapper = {}

    # Lower scalar float16s to pointers and dereference them
    for s in FindSymbols('scalars').visit(iet):
        if not np.issubdtype(s.dtype, np.float16) or s not in iet.parameters:
            continue

        # Replace the parameter with a pointer; replace occurences in the IET
        # body with a dereference (using the original symbol's dtype)
        ptr = s._rebuild(dtype=Float16P, is_const=True)
        val = Symbol(name=sregistry.make_name(prefix='hf'), dtype=s.dtype,
                     is_const=s.is_const)

        params_mapper[s], body_mapper[s] = ptr, val
        body_prefix.append(Dereference(val, ptr))  # val = *ptr

    # Apply the replacements
    body = body_prefix + as_list(Uxreplace(body_mapper).visit(iet.body))
    params = Uxreplace(params_mapper).visit(iet.parameters)

    iet = iet._rebuild(body=body, parameters=params)
    return iet, metadata


def _complex_includes(iet, lang, compiler):
    """
    Include complex arithmetic headers for the given language, if needed.
    """
    # Check if there is complex numbers that always take dtype precedence
    types = {f.dtype for f in FindSymbols().visit(iet)
             if not issubclass(f.dtype, ctypes._Pointer)}

    if not any(np.issubdtype(d, np.complexfloating) for d in types):
        return iet, {}

    metadata = {}
    lib = (lang['header-complex'],)

    if lang.get('complex-namespace') is not None:
        metadata['namespaces'] = lang['complex-namespace']

    # Some languges such as c++11 need some extra arithmetic definitions
    if lang.get('def-complex'):
        dest = compiler.get_jit_dir()
        hfile = dest.joinpath('complex_arith.h')
        with open(str(hfile), 'w') as ff:
            ff.write(str(lang['def-complex']))
        lib += (str(hfile),)

    metadata['includes'] = lib

    return iet, metadata
