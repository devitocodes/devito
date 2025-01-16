from sympy.printing.cxx import CXX11CodePrinter

from devito.ir import Call, UsingNamespace
from devito.passes.iet.langbase import LangBB
from devito.symbolics.printer import _DevitoPrinterBase
from devito.symbolics.extended_dtypes import c_complex, c_double_complex

__all__ = ['CXXBB']


std_arith = """
#include <complex>

template<typename _Tp, typename _Ti>
std::complex<_Tp> operator * (const _Ti & a, const std::complex<_Tp> & b){
  return std::complex<_Tp>(b.real() * a, b.imag() * a);
}

template<typename _Tp, typename _Ti>
std::complex<_Tp> operator * (const std::complex<_Tp> & b, const _Ti & a){
  return std::complex<_Tp>(b.real() * a, b.imag() * a);
}

template<typename _Tp, typename _Ti>
std::complex<_Tp> operator / (const _Ti & a, const std::complex<_Tp> & b){
  _Tp denom = b.real() * b.real () + b.imag() * b.imag()
  return std::complex<_Tp>(b.real() * a / denom, - b.imag() * a / denom);
}

template<typename _Tp, typename _Ti>
std::complex<_Tp> operator / (const std::complex<_Tp> & b, const _Ti & a){
  return std::complex<_Tp>(b.real() / a, b.imag() / a);
}

template<typename _Tp, typename _Ti>
std::complex<_Tp> operator + (const _Ti & a, const std::complex<_Tp> & b){
  return std::complex<_Tp>(b.real() + a, b.imag());
}

template<typename _Tp, typename _Ti>
std::complex<_Tp> operator + (const std::complex<_Tp> & b, const _Ti & a){
  return std::complex<_Tp>(b.real() + a, b.imag());
}

"""


class CXXBB(LangBB):

    mapper = {
        'header-memcpy': 'string.h',
        'host-alloc': lambda i, j, k:
            Call('posix_memalign', (i, j, k)),
        'host-alloc-pin': lambda i, j, k:
            Call('posix_memalign', (i, j, k)),
        'host-free': lambda i:
            Call('free', (i,)),
        'host-free-pin': lambda i:
            Call('free', (i,)),
        'alloc-global-symbol': lambda i, j, k:
            Call('memcpy', (i, j, k)),
        # Complex and float16
        'header-complex': 'complex',
        'complex-namespace': [UsingNamespace('std::complex_literals')],
        'def-complex': std_arith,
    }


class CXXDevitoPrinter(_DevitoPrinterBase, CXX11CodePrinter):

    _default_settings = {**_DevitoPrinterBase._default_settings,
                         **CXX11CodePrinter._default_settings}
    _ns = "std::"

    # These cannot go through _print_xxx because they are classes not
    # instances
    type_mappings = {**_DevitoPrinterBase.type_mappings,
                     c_complex: 'std::complex<float>',
                     c_double_complex: 'std::complex<float>',
                     **CXX11CodePrinter.type_mappings}

    def _print_ImaginaryUnit(self, expr):
        return f'1i{self.prec_literal(expr).lower()}'
