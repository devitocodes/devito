from devito.ir import Call, UsingNamespace
from devito.passes.iet.langbase import LangBB

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
