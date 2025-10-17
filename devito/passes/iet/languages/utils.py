import numpy as np

from devito import Real, Imag
from devito.exceptions import InvalidOperator
from devito.ir import List

__all__ = ['joins', '_atomic_add_split']


def joins(*symbols):
    return ",".join(sorted([i.name for i in symbols]))


def _atomic_add_split(i, pragmas, real, imag):
    # Complex reduction, split between real and imaginary parts.
    # real is a function i -> real(i)
    # imag is a function i -> imag(i)
    lhs, rhs = i.expr.lhs, i.expr.rhs
    if (np.issubdtype(lhs.dtype, np.complexfloating)
       and np.issubdtype(rhs.dtype, np.complexfloating)):
        # Complex i, complex j
        # Atomic add real and imaginary parts separately
        lhsr, rhsr = real(lhs), Real(rhs)
        lhsi, rhsi = imag(lhs), Imag(rhs)
        real = i._rebuild(expr=i.expr._rebuild(lhs=lhsr, rhs=rhsr),
                          pragmas=pragmas)
        imag = i._rebuild(expr=i.expr._rebuild(lhs=lhsi, rhs=rhsi),
                          pragmas=pragmas)
        return List(body=[real, imag])

    elif (np.issubdtype(lhs.dtype, np.complexfloating)
          and not np.issubdtype(rhs.dtype, np.complexfloating)):
        # Complex i, real j
        # Atomic add j to real part of i
        lhsr, rhsr = real(lhs), rhs
        real = i._rebuild(expr=i.expr._rebuild(lhs=lhsr, rhs=rhsr),
                          pragmas=pragmas)
        return real
    else:
        # Real i, complex j
        raise InvalidOperator("Atomic add not implemented for real "
                              "Functions with complex increments")
