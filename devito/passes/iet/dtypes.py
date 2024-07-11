import numpy as np
import ctypes

from devito.ir import FindSymbols, Uxreplace
from devito.ir.iet.nodes import Dereference
from devito.tools.utils import as_tuple
from devito.types.basic import Symbol

__all__ = ['lower_scalar_half', 'lower_complex']


def lower_scalar_half(iet, lang, sregistry):
    """
    Lower half float scalars to pointers (special case, since we can't
    pass them directly for lack of a ctypes equivalent)
    """
    if lang.get('half_types') is None:
        return iet, {}

    # dtype mappings for float16
    half, half_p = lang['half_types']

    body = []  # derefs to prepend to the body
    body_mapper = {}
    params_mapper = {}

    for s in FindSymbols('scalars').visit(iet):
        if s.dtype != np.float16 or s not in iet.parameters:
            continue

        ptr = s._rebuild(dtype=half_p)
        val = Symbol(name=sregistry.make_name(prefix='hf'), dtype=half, is_const=True)

        params_mapper[s] = ptr
        body_mapper[s] = val
        body.append(Dereference(val, ptr))  # val = *ptr

    body.extend(as_tuple(Uxreplace(body_mapper).visit(iet.body)))
    params = Uxreplace(params_mapper).visit(iet.parameters)

    iet = iet._rebuild(body=body, parameters=params)
    return iet, {}


def lower_complex(iet, lang, compiler):
    """
    Add headers for complex arithmetic
    """
    # Check if there is complex numbers that always take dtype precedence
    types = {f.dtype for f in FindSymbols().visit(iet)
             if not issubclass(f.dtype, ctypes._Pointer)}

    metadata = {}
    if any(np.issubdtype(d, np.complexfloating) for d in types):
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

    iet = _lower_dtypes(iet, lang)
    return iet, metadata


def _lower_dtypes(iet, lang):
    """
    Lower dtypes to language specific types
    """
    mapper = {}

    for s in FindSymbols('indexeds|basics|symbolics').visit(iet):
        if s.dtype in lang['types']:
            mapper[s] = s._rebuild(dtype=lang['types'][s.dtype])

    body = Uxreplace(mapper).visit(iet.body)
    params = Uxreplace(mapper).visit(iet.parameters)
    iet = iet._rebuild(body=body, parameters=params)

    return iet
