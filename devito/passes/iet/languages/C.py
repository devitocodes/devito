import numpy as np
from sympy.printing.c import C99CodePrinter

from devito.exceptions import InvalidOperator
from devito.ir import Call, BasePrinter, List
from devito.passes.iet.definitions import DataManager
from devito.passes.iet.orchestration import Orchestrator
from devito.passes.iet.langbase import LangBB
from devito.symbolics import c_complex, c_double_complex
from devito.symbolics.extended_sympy import UnaryOp
from devito.tools import dtype_to_cstr

__all__ = ['CBB', 'CDataManager', 'COrchestrator']


class RealExt(UnaryOp):

    _op = '__real__ '


class ImagExt(UnaryOp):

    _op = '__imag__ '


def atomic_add(i, pragmas, split=False):
    # Base case, real reduction
    if not split:
        return i._rebuild(pragmas=pragmas)
    # Complex reduction, split using a temp pointer
    # Transforns lhs += rhs into
    # {
    #   pragmas
    #   __real__ lhs += __real__ rhs;
    #   pragmas
    #   __imag__ lhs += __imag__ rhs;
    # }
    lhs, rhs = i.expr.lhs, i.expr.rhs
    if (np.issubdtype(lhs.dtype, np.complexfloating)
       and np.issubdtype(rhs.dtype, np.complexfloating)):
        # Complex i, complex j
        # Atomic add real and imaginary parts separately
        lhsr, rhsr = RealExt(lhs), RealExt(rhs)
        lhsi, rhsi = ImagExt(lhs), ImagExt(rhs)
        real = i._rebuild(expr=i.expr._rebuild(lhs=lhsr, rhs=rhsr),
                          pragmas=pragmas)
        imag = i._rebuild(expr=i.expr._rebuild(lhs=lhsi, rhs=rhsi),
                          pragmas=pragmas)
        return List(body=[real, imag])

    elif (np.issubdtype(lhs.dtype, np.complexfloating)
          and not np.issubdtype(rhs.dtype, np.complexfloating)):
        # Complex i, real j
        # Atomic add j to real part of i
        lhsr, rhsr = RealExt(lhs), rhs
        real = i._rebuild(expr=i.expr._rebuild(lhs=lhsr, rhs=rhsr),
                          pragmas=pragmas)
        return real
    else:
        # Real i, complex j
        raise InvalidOperator("Atomic add not implemented for real "
                              "Functions with complex increments")


class CBB(LangBB):

    mapper = {
        # Misc
        'header-array': None,
        # Complex
        'includes-complex': 'complex.h',
        # Allocs
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
            Call('memcpy', (i, j, k))
    }


class CDataManager(DataManager):
    langbb = CBB


class COrchestrator(Orchestrator):
    langbb = CBB


class CPrinter(BasePrinter, C99CodePrinter):

    _default_settings = {**BasePrinter._default_settings,
                         **C99CodePrinter._default_settings}
    _func_literals = {np.float32: 'f', np.complex64: 'f'}
    _func_prefix = {np.float32: 'f', np.float64: 'f',
                    np.complex64: 'c', np.complex128: 'c'}
    _includes = ['stdlib.h', 'math.h', 'sys/time.h']

    # These cannot go through _print_xxx because they are classes not
    # instances
    type_mappings = {**C99CodePrinter.type_mappings,
                     c_complex: 'float _Complex',
                     c_double_complex: 'double _Complex'}

    def _print_ImaginaryUnit(self, expr):
        return '_Complex_I'

    def _print_ListInitializer(self, expr):
        li = super()._print_ListInitializer(expr)
        if expr.dtype:
            # C99, unlike CXX, supports compound literals
            tstr = dtype_to_cstr(expr.dtype)
            return f'({tstr}[]){li}'
        else:
            return li

    def _print_ComplexPart(self, expr):
        return (f'{self.func_prefix(expr)}{expr._name}{self.func_literal(expr)}'
                f'({self._print(expr.args[0])})')

    def _print_Conj(self, expr):
        # In C, conj is not preceeded by the func_prefix
        return (f'conj{self.func_literal(expr)}({self._print(expr.args[0])})')
