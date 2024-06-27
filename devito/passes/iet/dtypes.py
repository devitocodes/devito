import numpy as np
import ctypes

from devito.ir import FindSymbols, Uxreplace

__all__ = ['lower_complex']


def lower_complex(iet, lang, compiler):
    """
    Add headers for complex arithmetic
    """
    # Check if there is complex numbers that always take dtype precedence
    types = {f.dtype for f in FindSymbols().visit(iet)
             if not issubclass(f.dtype, ctypes._Pointer)}

    if not any(np.issubdtype(d, np.complexfloating) for d in types):
        return iet, {}

    lib = (lang['header-complex'],)

    metadata = {}
    if lang.get('complex-namespace') is not None:
        metadata['namespaces'] = lang['complex-namespace']

    # Some languges such as c++11 need some extra arithmetic definitions
    if lang.get('def-complex'):
        dest = compiler.get_jit_dir()
        hfile = dest.joinpath('complex_arith.h')
        with open(str(hfile), 'w') as ff:
            ff.write(str(lang['def-complex']))
        lib += (str(hfile),)

    iet = _complex_dtypes(iet, lang)
    metadata['includes'] = lib

    return iet, metadata


def _complex_dtypes(iet, lang):
    """
    Lower dtypes to language specific types
    """
    mapper = {}

    for s in FindSymbols('indexeds').visit(iet):
        if s.dtype in lang['types']:
            mapper[s] = s._rebuild(dtype=lang['types'][s.dtype])

    for s in FindSymbols().visit(iet):
        if s.dtype in lang['types']:
            mapper[s] = s._rebuild(dtype=lang['types'][s.dtype])

    body = Uxreplace(mapper).visit(iet.body)
    params = Uxreplace(mapper).visit(iet.parameters)
    iet = iet._rebuild(body=body, parameters=params)

    return iet
