import numpy as np
from sympy.printing.cxx import CXX11CodePrinter

from devito.ir import Call, UsingNamespace, BasePrinter
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB
from devito.passes.iet.languages.utils import _atomic_add_split
from devito.symbolics import c_complex, c_double_complex, IndexedPointer, cast, Byref
from devito.tools import dtype_to_cstr

__all__ = ['CXXBB', 'CXXDataManager', 'CXXOrchestrator']


def std_arith(prefix=None):
    if prefix:
        # Method definition prefix, e.g. "__host__"
        # Make sure there is a space between the prefix and the method name
        prefix = prefix if prefix.endswith(" ") else f"{prefix} "
    else:
        prefix = ""
    return f"""
#include <complex>

template<typename _Tp, typename _Ti>
{prefix}std::complex<_Tp> operator * (const _Ti & a, const std::complex<_Tp> & b){{
  return std::complex<_Tp>(b.real() * a, b.imag() * a);
}}

template<typename _Tp, typename _Ti>
{prefix}std::complex<_Tp> operator * (const std::complex<_Tp> & b, const _Ti & a){{
  return std::complex<_Tp>(b.real() * a, b.imag() * a);
}}

template<typename _Tp, typename _Ti>
{prefix}std::complex<_Tp> operator / (const _Ti & a, const std::complex<_Tp> & b){{
  _Tp denom = b.real() * b.real () + b.imag() * b.imag();
  return std::complex<_Tp>(b.real() * a / denom, - b.imag() * a / denom);
}}

template<typename _Tp, typename _Ti>
{prefix}std::complex<_Tp> operator / (const std::complex<_Tp> & b, const _Ti & a){{
  return std::complex<_Tp>(b.real() / a, b.imag() / a);
}}

template<typename _Tp, typename _Ti>
{prefix}std::complex<_Tp> operator + (const _Ti & a, const std::complex<_Tp> & b){{
  return std::complex<_Tp>(b.real() + a, b.imag());
}}

template<typename _Tp, typename _Ti>
{prefix}std::complex<_Tp> operator + (const std::complex<_Tp> & b, const _Ti & a){{
  return std::complex<_Tp>(b.real() + a, b.imag());
}}

template<typename _Tp, typename _Ti>
{prefix}std::complex<_Tp> operator - (const _Ti & a, const std::complex<_Tp> & b){{
  return std::complex<_Tp>(a = b.real(), b.imag());
}}

template<typename _Tp, typename _Ti>
{prefix}std::complex<_Tp> operator - (const std::complex<_Tp> & b, const _Ti & a){{
  return std::complex<_Tp>(b.real() - a, b.imag());
}}

"""


def split_pointer(i, idx):
    """
    Splits complex pointer std::complex<T> *a as
    (float *)(&a)[idx] for real/imag parts.
    """
    dtype = i.dtype(0).real.__class__
    ptr = cast(dtype, stars='*')(Byref(i), reinterpret=True)
    return IndexedPointer(ptr, idx)


cxx_imag = lambda i: split_pointer(i, 1)
cxx_real = lambda i: split_pointer(i, 0)


def atomic_add(i, pragmas, split=False):
    # Base case, real reduction
    if not split:
        return i._rebuild(pragmas=pragmas)

    # Complex reduction, split using a temp pointer
    # Transforms lhs += rhs into
    # {
    #   pragmas
    #   reinterpret_cast<float*>(&lhs)[0] += std::real(rhs);
    #   pragmas
    #   reinterpret_cast<float*>(&lhs)[1] += std::imag(rhs);
    # }
    # Make a temp pointer
    return _atomic_add_split(i, pragmas, cxx_real, cxx_imag)


class CXXBB(LangBB):

    mapper = {
        # Misc
        'header-array': 'array',
        # Complex
        'includes-complex': 'complex',
        'complex-namespace': [UsingNamespace('std::complex_literals')],
        'def-complex': std_arith(),
        # Allocs
        'header-memcpy': 'string.h',
        'header-math': 'algorithm',
        'host-alloc': lambda i, j, k:
            Call('posix_memalign', (i, j, k)),
        'host-alloc-pin': lambda i, j, k:
            Call('posix_memalign', (i, j, k)),
        'host-free': lambda i:
            Call('free', (i,)),
        'host-free-pin': lambda i:
            Call('free', (i,)),
        'alloc-global-symbol': lambda i, j, k:
            Call('memcpy', (i, j, k))
    }


class CXXDataManager(DataManager):
    langbb = CXXBB


class CXXOrchestrator(Orchestrator):
    langbb = CXXBB


class CXXPrinter(BasePrinter, CXX11CodePrinter):

    _default_settings = {**BasePrinter._default_settings,
                         **CXX11CodePrinter._default_settings}
    _ns = "std::"
    _func_literals = {}
    _func_prefix = {np.float32: 'f', np.float64: 'f'}
    _restrict_keyword = '__restrict'
    _includes = ['cstdlib', 'cmath', 'sys/time.h']

    # These cannot go through _print_xxx because they are classes not
    # instances
    type_mappings = {**CXX11CodePrinter.type_mappings,
                     c_complex: 'std::complex<float>',
                     c_double_complex: 'std::complex<double>'}

    def _print_ImaginaryUnit(self, expr):
        return f'1i{self.prec_literal(expr).lower()}'

    def _print_ComplexPart(self, expr):
        return f'{self._ns}{expr._name}({self._print(expr.args[0])})'

    def _print_Cast(self, expr):
        # The CXX recommended way to cast is to use static_cast
        tstr = self._print(expr._C_ctype)
        if 'void' in tstr:
            return super()._print_Cast(expr)
        caster = 'reinterpret_cast' if expr.reinterpret else 'static_cast'
        cast = f'{caster}<{tstr}{self._print(expr.stars)}>'
        return self._print_UnaryOp(expr, op=cast, parenthesize=True)

    def _print_ListInitializer(self, expr):
        li = super()._print_ListInitializer(expr)
        if expr.dtype:
            # CXX, unlike C99, does not support compound literals
            tstr = dtype_to_cstr(expr.dtype)
            return f'std::array<{tstr}, {len(expr.params)}>{li}.data()'
        else:
            return li
