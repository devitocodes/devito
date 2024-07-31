import numpy as np

from devito.ir import Call, UsingNamespace
from devito.passes.iet.langbase import LangBB
from devito.passes.iet.languages.C import c_float16, c_float16_p
from devito.symbolics.extended_dtypes import Float16P, c_complex, c_double_complex

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


cxx_complex = type('std::complex<float>', (c_complex,), {})
cxx_double_complex = type('std::complex<double>', (c_double_complex,), {})


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
        "types": {np.complex128: cxx_double_complex,
                  np.complex64: cxx_complex,
                  np.float16: c_float16,
                  Float16P: c_float16_p}
    }
