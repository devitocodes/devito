import numpy as np
import ctypes

from devito.ir import FindSymbols, Uxreplace
from devito.ir.iet.nodes import Dereference
from devito.tools.utils import as_list
from devito.types.basic import Symbol

__all__ = ['lower_dtypes']


def lower_dtypes(iet, lang, compiler, sregistry):
    """
    Lower language-specific dtypes and add headers for complex arithmetic
    """
    # Include complex headers if needed (before we replace complex dtypes)
    metadata = _complex_includes(iet, lang, compiler)

    body_prefix = []  # Derefs to prepend to the body
    body_mapper = {}
    params_mapper = {}

    # Lower scalar float16s to pointers and dereference them
    if lang.get('half_types') is not None:
        half, half_p = lang['half_types']  # dtype mappings for half float

        for s in FindSymbols('scalars').visit(iet):
            if s.dtype != np.float16 or s not in iet.parameters:
                continue

            ptr = s._rebuild(dtype=half_p, is_const=True)
            val = Symbol(name=sregistry.make_name(prefix='hf'), dtype=half,
                         is_const=s.is_const)

            params_mapper[s], body_mapper[s] = ptr, val
            body_prefix.append(Dereference(val, ptr))  # val = *ptr

    # Lower remaining language-specific dtypes
    for s in FindSymbols('indexeds|basics|symbolics').visit(iet):
        if s.dtype in lang['types'] and s not in params_mapper:
            body_mapper[s] = params_mapper[s] = s._rebuild(dtype=lang['types'][s.dtype])

    # Apply the dtype replacements
    body = body_prefix + as_list(Uxreplace(body_mapper).visit(iet.body))
    params = Uxreplace(params_mapper).visit(iet.parameters)

    iet = iet._rebuild(body=body, parameters=params)
    return iet, metadata


def _complex_includes(iet, lang, compiler):
    """
    Add headers for complex arithmetic
    """
    # Check if there is complex numbers that always take dtype precedence
    types = {f.dtype for f in FindSymbols().visit(iet)
             if not issubclass(f.dtype, ctypes._Pointer)}

    if not any(np.issubdtype(d, np.complexfloating) for d in types):
        return {}

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

    return metadata
