from sympy import Rational, sympify

__all__ = ['Closure']


class Closure:

    """
    The finite differences along a Dimension whose stencils vary near a boundary,
    such as summation-by-parts closures, by where their points are.

    A Closure is given as the weights of a derivative along its Dimension, e.g.
    `u.dz(weights={z: closure})`, and it gives the stencil of the finite difference
    from the points of its argument to its `x0`, or for a transpose, of the one it
    transposes. Along a Dimension it does not differentiate, it gives the stencil
    of the interpolation. A pair of points without a stencil takes Taylor weights.

    Parameters
    ----------
    dim : Dimension
        The Dimension of the finite differences.
    stencils : dict
        For each `(source, target, deriv_order)`, the offsets from the nodes of the
        source and the target points in grid spacings, 0 or 1/2, and the order of
        the derivative, 0 for the interpolation: the offsets of the stencil points
        from the target in grid spacings, and the Function of their weights per
        unit spacing, of `dim` and a stencil Dimension, which takes at each target
        point the weights of its row.
    """

    def __init__(self, dim, stencils):
        self._dim = dim
        self._stencils = {(Rational(s), Rational(t), order): stencil
                          for (s, t, order), stencil in stencils.items()}

    def __repr__(self):
        return f"Closure({self._dim})"

    @property
    def dim(self):
        """The Dimension of the finite differences."""
        return self._dim

    def stencil(self, source, target, deriv_order):
        """
        Return the offsets of the stencil points and their weights per unit
        spacing, from the `source` points to the `target` points (see `stencils`),
        None without one.
        """
        return self._stencils.get((_offset(source, self._dim),
                                   _offset(target, self._dim), deriv_order))


def _offset(point, dim):
    """The offset of `point` from the nodes along `dim`, in grid spacings, mod 1."""
    return sympify((point - dim)/dim.spacing) % 1
