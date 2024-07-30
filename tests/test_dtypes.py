import numpy as np
import pytest
import sympy

from devito import Constant, Eq, Function, Grid, Operator
from devito.ir.iet.nodes import Dereference
from devito.passes.iet.langbase import LangBB
from devito.passes.iet.languages.C import CBB
from devito.passes.iet.languages.openacc import AccBB
from devito.passes.iet.languages.openmp import OmpBB
from devito.symbolics.extended_dtypes import Float16P
from devito.types.basic import Basic, Scalar, Symbol
from devito.types.dimension import Dimension, Spacing

# Mappers for language-specific types and headers
_languages: dict[str, type[LangBB]] = {
    'C': CBB,
    'openmp': OmpBB,
    'openacc': AccBB
}


def _get_language(language: str, **_) -> type[LangBB]:
    """
    Gets the language building block type from parametrized kwargs.
    """

    return _languages[language]


def _config_kwargs(platform: str, language: str, compiler: str) -> dict[str, str]:
    """
    Generates kwargs for Operator to test language-specific behavior.
    """

    return {
        'platform': platform,
        'language': language,
        'compiler': compiler
    }


# List of pararmetrized operator kwargs for testing language-specific behavior
_configs: list[dict[str, str]] = [
    _config_kwargs(*cfg) for cfg in [
        ('cpu64', 'C', 'gcc'),
        ('cpu64', 'openmp', 'gcc'),
        ('nvidiaX', 'openmp', 'nvc'),
        ('nvidiaX', 'openacc', 'nvc')
    ]
]


@pytest.mark.parametrize('dtype', [np.float16, np.complex64, np.complex128])
@pytest.mark.parametrize('kwargs', _configs)
def test_dtype_mapping(dtype: np.dtype, kwargs: dict[str, str]) -> None:
    """
    Tests that half and complex floats' dtypes result in the correct type
    strings in generated code.
    """

    # Retrieve the language-specific type mapping
    lang_types: dict[np.dtype, type] = _get_language(**kwargs).get('types')

    # Set up an operator
    grid = Grid(shape=(3, 3))
    x: Dimension
    y: Dimension
    x, y = grid.dimensions

    c = Constant(name='c', dtype=dtype)
    u = Function(name='u', grid=grid, dtype=dtype)
    eq = Eq(u, c * x * y)
    op = Operator(eq, **kwargs)

    # Check ctypes of the mapped parameters
    params: dict[str, Basic] = {p.name: p for p in op.parameters}
    _u: Function = params['u']
    _c: Constant = params['c']
    assert _u.indexed._C_ctype._type_ == lang_types[_u.dtype]
    assert _c._C_ctype == lang_types[_c.dtype]


def test_half_params() -> None:
    """
    Tests float16 input parameters: scalars should be lowered to pointers
    and dereferenced; other parameters should keep the original dtype.
    """

    grid = Grid(shape=(5, 5), dtype=np.float16)
    x: Dimension
    y: Dimension
    x, y = grid.dimensions

    c = Constant(name='c', dtype=np.float16)
    u = Function(name='u', grid=grid)
    eq = Eq(u, x * x.spacing + c * y * y.spacing)
    op = Operator(eq)

    # Check that lowered parameters have the correct dtypes
    params: dict[str, Basic] = {p.name: p for p in op.parameters}
    _u: Function = params['u']
    _c: Constant = params['c']
    _dx: Spacing = params['h_x']
    _dy: Spacing = params['h_y']

    assert _u.dtype == np.float16
    assert _c.dtype == Float16P
    assert _dx.dtype == Float16P
    assert _dy.dtype == Float16P

    # Ensure the mapped pointer-to-half symbols are dereferenced
    derefs: set[Symbol] = {n.pointer for n in op.body.body
                           if isinstance(n, Dereference)}
    assert _c in derefs
    assert _dx in derefs
    assert _dy in derefs


@pytest.mark.parametrize('dtype', [np.float16, np.float32,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize('kwargs', _configs)
def test_complex_headers(dtype: np.dtype, kwargs: dict[str, str]) -> None:
    """
    Tests that the correct complex headers are included when complex dtypes
    are present in the operator, and omitted otherwise.
    """

    # Set up an operator
    grid = Grid(shape=(3, 3))
    x: Dimension
    y: Dimension
    x, y = grid.dimensions

    c = Constant(name='c', dtype=dtype)
    u = Function(name='u', grid=grid, dtype=dtype)
    eq = Eq(u, c * x * y)
    op = Operator(eq, **kwargs)

    # Check that the complex header is included <=> complex dtypes are present
    header: str = _get_language(**kwargs).get('header-complex')
    if np.issubdtype(dtype, np.complexfloating):
        assert header in op._includes
    else:
        assert header not in op._includes


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('kwargs', _configs)
def test_imag_unit(dtype: np.complexfloating, kwargs: dict[str, str]) -> None:
    """
    Tests that the correct literal is used for the imaginary unit.
    """

    # Determine the expected imaginary unit string
    unit_str: str
    if kwargs['compiler'] == 'gcc':
        # In C we multiply by the _Complex_I macro constant
        unit_str = '_Complex_I'
    else:
        # C++ provides an imaginary literal
        unit_str = '1if' if dtype == np.complex64 else '1i'

    # Set up an operator
    s = Symbol(name='s', dtype=dtype)
    eq = Eq(s, 2.0 + 3.0j)
    op = Operator(eq, **kwargs)

    # Check that the correct imaginary unit is used
    assert unit_str in str(op)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize(['sym', 'fun'], [(sympy.exp, np.exp),
                                          (sympy.log, np.log),
                                          (sympy.sin, np.sin)])
def test_math_functions(dtype: np.dtype, sym: sympy.Function, fun: np.ufunc) -> None:
    """
    Tests that the correct math functions are used, and their results cast
    and assigned appropriately for different float precisions and for
    complex floats/doubles.
    """

    # Get the expected function call string
    call_str = str(sym)
    if np.issubdtype(dtype, np.complexfloating):
        # Complex functions have a 'c' prefix
        call_str = 'c%s' % call_str
    if dtype(0).real.itemsize <= 4:
        # Single precision have an 'f' suffix (half is promoted to single)
        call_str = '%sf' % call_str

    # Operator setup
    a = Symbol(name='a', dtype=dtype)
    b = Scalar(name='b', dtype=dtype)

    eq = Eq(a, sym(b))
    op = Operator(eq)

    # Ensure the generated function call has the correct form
    assert call_str in str(op)
