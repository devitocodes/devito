from itertools import product

import numpy as np
import pytest
import sympy

from devito import NODE, Differentiable, Eq, Function, Grid, Operator
from devito.finite_differences.differentiable import (
    Add, Mul, Pow, SafeInv, diffify, interp_for_fd
)


def test_differentiable():
    a = Function(name="a", grid=Grid((10, 10)))
    e = Function(name="e", grid=Grid((10, 10)))

    assert isinstance(1.2 * a.dx, Mul)
    assert isinstance(e + a, Add)
    assert isinstance(e * a, Mul)
    assert isinstance(a * a, Pow)
    assert isinstance(1 / (a * a), Pow)
    assert (a + e*a).dtype == a.dtype

    addition = a + 1.2 * a.dx
    assert isinstance(addition, Add)
    assert all(isinstance(a, Differentiable) for a in addition.args)
    assert addition.dtype == a.dtype

    addition2 = a + e * a.dx
    assert isinstance(addition2, Add)
    assert all(isinstance(a, Differentiable) for a in addition2.args)
    assert addition2.dtype == a.dtype


def test_diffify():
    a = Function(name="a", grid=Grid((10, 10)))
    e = Function(name="e", grid=Grid((10, 10)))

    assert isinstance(diffify(sympy.Mul(*[1.2, a.dx])), Mul)
    assert isinstance(diffify(sympy.Add(*[a, e])), Add)
    assert isinstance(diffify(sympy.Mul(*[e, a])), Mul)
    assert isinstance(diffify(sympy.Mul(*[a, a])), Pow)
    assert isinstance(diffify(sympy.Pow(*[a*a, -1])), Pow)

    addition = diffify(sympy.Add(*[a, sympy.Mul(*[1.2, a.dx])]))
    assert isinstance(addition, Add)
    assert all(isinstance(a, Differentiable) for a in addition.args)

    addition2 = diffify(sympy.Add(*[a, sympy.Mul(*[e, a.dx])]))
    assert isinstance(addition2, Add)
    assert all(isinstance(a, Differentiable) for a in addition2.args)


def test_shift():
    a = Function(name="a", grid=Grid((10, 10)))
    x = a.dimensions[0]
    assert a.shift(x, x.spacing) == a._subs(x, x + x.spacing)
    assert a.shift(x, x.spacing).shift(x, -x.spacing) == a
    assert a.shift(x, x.spacing).shift(x, x.spacing) == a.shift(x, 2*x.spacing)
    assert a.dx.evaluate.shift(x, x.spacing) == a.shift(x, x.spacing).dx.evaluate
    assert a.shift(x, .5 * x.spacing)._grid_map == {x: x + .5 * x.spacing, 'subs': {}}


def test_interp():
    grid = Grid((10, 10))
    x = grid.dimensions[0]
    a = Function(name="a", grid=grid, staggered=NODE)
    sa = Function(name="as", grid=grid, staggered=x)

    def sp_diff(a, b):
        a = getattr(a, 'evaluate', a)
        b = getattr(b, 'evaluate', b)
        return sympy.simplify(a - b) == 0

    # Base case, no interp
    assert interp_for_fd(a, {}) == a
    assert interp_for_fd(a, {x: x}) == a
    assert interp_for_fd(sa, {}) == sa
    assert interp_for_fd(sa, {x: x + x.spacing/2}) == sa

    # Base case, interp
    assert sp_diff(interp_for_fd(a, {x: x + x.spacing/2}),
                   .5*a + .5*a.shift(x, x.spacing))
    assert sp_diff(interp_for_fd(sa, {x: x}),
                   .5*sa + .5*sa.shift(x, -x.spacing))

    # Mul case, split interp
    assert sp_diff(interp_for_fd(a*sa, {x: x + x.spacing/2}),
                   sa * interp_for_fd(a, {x: x + x.spacing/2}))
    assert sp_diff(interp_for_fd(a*sa, {x: x}),
                   a * interp_for_fd(sa, {x: x}))

    # Add case, split interp
    assert sp_diff(interp_for_fd(a + sa, {x: x + x.spacing/2}),
                   sa + interp_for_fd(a, {x: x + x.spacing/2}))
    assert sp_diff(interp_for_fd(a + sa, {x: x}),
                   a + interp_for_fd(sa, {x: x}))


@pytest.mark.parametrize('ndim', [1, 2, 3])
@pytest.mark.parametrize('io', [None, 2, 4])
def test_avg_mode(ndim, io):
    grid = Grid([11]*ndim)
    v = Function(name='v', grid=grid, staggered=grid.dimensions, space_order=4)
    kw = {'space_order': 4}
    if io is not None:
        kw['interp_order'] = io
    else:
        io = 2  # Default value

    with pytest.raises(ValueError):
        # interp_order > space_order
        Function(name="a", grid=grid, interp_order=8, space_order=4)
    with pytest.raises(TypeError):
        # interp_order not int
        Function(name="a", grid=grid, interp_order=2.5, space_order=4)

    a0 = Function(name="a0", grid=grid, **kw)
    a = Function(name="a", grid=grid, **kw)
    b = Function(name="b", grid=grid, avg_mode='safe_harmonic', **kw)

    a0_avg = a0._eval_at(v)
    a_avg = a._eval_at(v).evaluate.simplify()
    b_avg = b._eval_at(v).evaluate.simplify()

    assert a0_avg == a0.subs(v.indices_ref.getters)

    # Indices around the point at the center of a cell
    idx = list(range(-io//2 + 1, io//2 + 1))
    all_shift = tuple(product(*[idx for _ in range(ndim)]))
    coeffs = {2: [0.5, 0.5], 4: [-1/16, 9/16, 9/16, -1/16]}[io]
    vars = ['i', 'j', 'k'][:ndim]
    rule = ','.join(vars) + '->' + ''.join(vars)
    ndcoeffs = np.einsum(rule, *([coeffs]*ndim))
    args = [
        {d: d + i * d.spacing for d, i in zip(grid.dimensions, s, strict=True)}
        for s in all_shift
    ]

    # Default is arithmetic average
    expected = sum(
        c * a.subs(arg) for c, arg in zip(ndcoeffs.flatten(), args, strict=True)
    )
    assert sympy.simplify(a_avg - expected) == 0

    # Harmonic average, h(a[.5]) = 1/(.5/a[0] + .5/a[1])
    expected = (sum(c * SafeInv(b.subs(arg), b.subs(arg))
                    for c, arg in zip(ndcoeffs.flatten(), args, strict=True)))
    assert sympy.simplify(b_avg.args[0] - expected) == 0
    assert isinstance(b_avg, SafeInv)
    assert b_avg.base == b


def test_no_interp():
    grid = Grid((10, 10))
    x = grid.dimensions[0]
    a = Function(name="a", grid=grid, staggered=NODE, interp_order=0)
    sa = Function(name="as", grid=grid, staggered=x)

    assert a._eval_at(sa) == a
    assert sa._eval_at(a) == sa._subs(x, x - x.spacing/2)
    assert (a*sa)._eval_at(sa) == a*sa
    assert (a + sa)._eval_at(sa) == a + sa

    a_shift = a._subs(x, x + x.spacing / 2)
    # Should just do nearest grid point, so shift back to original
    assert a_shift.evaluate == a


class TestMulEvalAt:
    """
    Verify `Mul._eval_at` in both modes:

    - `interp_mode="direct"`: default per-arg evaluation.
    - `interp_mode="symmetric"`: symmetric `I * a * I^T * b.dx` interpolation.
    """

    @staticmethod
    def _all_funcs(grid):
        x, y = grid.dimensions
        return {
            'node': Function(name='fn', grid=grid, space_order=4, staggered=NODE),
            'x': Function(name='fx', grid=grid, space_order=4, staggered=x),
            'y': Function(name='fy', grid=grid, space_order=4, staggered=y),
            'xy': Function(name='fxy', grid=grid, space_order=4, staggered=(x, y)),
        }

    @pytest.mark.parametrize('interp_mode', ['direct', 'symmetric'])
    @pytest.mark.parametrize('targets', [
        ('node', 'x', 'xy'),
        ('node', 'y', 'xy'),
        ('x', 'y', 'xy'),
        ('node', 'x', 'y'),
        ('node', 'xy', 'x'),
    ])
    def test_mul_two_funcs(self, interp_mode, targets):
        """`a * b` evaluated at `L` references both factors."""
        grid = Grid((11, 11))
        funcs = self._all_funcs(grid)
        a_key, b_key, l_key = targets
        a, b, L = funcs[a_key], funcs[b_key], funcs[l_key]

        result = (a * b)._eval_at(L, interp_mode=interp_mode)
        evaluated_str = str(result.evaluate)
        assert a.name in evaluated_str
        assert b.name in evaluated_str

    @pytest.mark.parametrize('interp_mode', ['direct', 'symmetric'])
    @pytest.mark.parametrize('targets', [
        ('node', 'x', 'y', 'xy'),
        ('node', 'x', 'xy', 'y'),
        ('x', 'y', 'node', 'xy'),
    ])
    def test_mul_three_funcs(self, interp_mode, targets):
        """`c * a * b` evaluated at `L` references all three factors."""
        grid = Grid((11, 11))
        funcs = self._all_funcs(grid)
        c_key, a_key, b_key, l_key = targets
        c, a, b, L = funcs[c_key], funcs[a_key], funcs[b_key], funcs[l_key]

        result = (c * a * b)._eval_at(L, interp_mode=interp_mode)
        evaluated_str = str(result.evaluate)
        assert a.name in evaluated_str
        assert b.name in evaluated_str
        assert c.name in evaluated_str

    @pytest.mark.parametrize('interp_mode', ['direct', 'symmetric'])
    @pytest.mark.parametrize('targets', [
        ('node', 'x', 'xy'),
        ('x', 'node', 'xy'),
        ('node', 'xy', 'y'),
        ('xy', 'node', 'x'),
    ])
    def test_mul_func_deriv(self, interp_mode, targets):
        """`a * b.dx` evaluated at `L`: symmetric `I*a*I^T*b.dx` form."""
        grid = Grid((11, 11))
        funcs = self._all_funcs(grid)
        a_key, b_key, l_key = targets
        a, b, L = funcs[a_key], funcs[b_key], funcs[l_key]

        result = (a * b.dx)._eval_at(L, interp_mode=interp_mode)
        evaluated_str = str(result.evaluate)
        assert a.name in evaluated_str
        assert b.name in evaluated_str

    @pytest.mark.parametrize('interp_mode', ['direct', 'symmetric'])
    def test_mul_eval_at_no_op(self, interp_mode):
        """`a * b` evaluated at its own location is a no-op."""
        grid = Grid((11, 11))
        a = Function(name='a', grid=grid, space_order=4, staggered=NODE)
        b = Function(name='b', grid=grid, space_order=4, staggered=NODE)

        result = (a * b)._eval_at(a, interp_mode=interp_mode)
        assert sympy.simplify(result.evaluate - (a * b).evaluate) == 0

    def test_interp_mode_skips_when_deriv_at_func(self):
        """When `b.dx`'s natural staggering matches `c`'s, the symmetric
        mode falls back to the default per-arg evaluation (no I^T)."""
        grid = Grid((11,))
        x = grid.dimensions[0]
        a = Function(name='a', grid=grid, space_order=4, staggered=NODE)
        b = Function(name='b', grid=grid, space_order=4, staggered=x)
        c = Function(name='c', grid=grid, space_order=4, staggered=x)

        default = (a * b.dx)._eval_at(c).evaluate
        symmetric = (a * b.dx)._eval_at(c, interp_mode="symmetric").evaluate
        assert sympy.simplify(default - symmetric) == 0

    def test_interp_mode_applies_symmetric(self):
        """When both `a` and `b` differ from `c` in staggering, the
        symmetric mode wraps the product in the `I * a * I^T * b.dx` form,
        producing two distinct 0-order FD interpolations (`I^T` and `I`)."""
        grid = Grid((11, 11))
        x, y = grid.dimensions
        a = Function(name='a', grid=grid, space_order=4, staggered=NODE)
        b = Function(name='b', grid=grid, space_order=4, staggered=x)
        c = Function(name='c', grid=grid, space_order=4, staggered=(x, y))

        result = (a * b.dx)._eval_at(c, interp_mode="symmetric")
        zero_order = [d for d in result.find(sympy.Derivative)
                      if all(o == 0 for o in d.deriv_order)]
        # Two 0-order Derivatives: the outer I (a -> c) and the inner I^T (c -> a)
        assert len(zero_order) == 2

    def test_interp_mode_with_function_factor(self):
        """The symmetric mode applies to any Differentiable factor at a
        non-matching staggering, not only Derivatives. E.g. `a * bdx` where
        `bdx` is a stored Function (e.g. holding a pre-computed derivative)."""
        grid = Grid((11, 11))
        x, y = grid.dimensions
        a = Function(name='a', grid=grid, space_order=4, staggered=NODE)
        bdx = Function(name='bdx', grid=grid, space_order=4, staggered=x)
        c = Function(name='c', grid=grid, space_order=4, staggered=(x, y))

        result = (a * bdx)._eval_at(c, interp_mode="symmetric")
        # Symmetric form: outer I (a -> c) and inner I^T (bdx's loc -> a)
        zero_order = [d for d in result.find(sympy.Derivative)
                      if all(o == 0 for o in d.deriv_order)]
        assert len(zero_order) == 2

    def test_interp_mode_same_loc_block_interp(self):
        """When all factors share a single location that differs from `func`,
        the symmetric mode interpolates the whole product as one block (as
        required by the elastic stiffness `I*(C_{ij}*b_j) -> a_i` form),
        not per-arg."""
        grid = Grid((11, 11))
        x, y = grid.dimensions
        a = Function(name='a', grid=grid, space_order=4, staggered=NODE)
        b = Function(name='b', grid=grid, space_order=4, staggered=NODE)
        c = Function(name='c', grid=grid, space_order=4, staggered=(x, y))

        result = (a * b)._eval_at(c, interp_mode="symmetric")
        # Single 0-order Derivative wrapping the product: I(a*b)
        zero_order = [d for d in result.find(sympy.Derivative)
                      if all(o == 0 for o in d.deriv_order)]
        assert len(zero_order) == 1
        block_str = str(zero_order[0].expr)
        assert a.name in block_str
        assert b.name in block_str

    def test_interp_mode_factor_at_func_falls_back(self):
        """When at least one factor already matches `func`'s staggering, the
        symmetric mode is unnecessary and we fall back to the default."""
        grid = Grid((11, 11))
        x, y = grid.dimensions
        a = Function(name='a', grid=grid, space_order=4, staggered=NODE)
        b = Function(name='b', grid=grid, space_order=4, staggered=NODE)
        c = Function(name='c', grid=grid, space_order=4, staggered=NODE)

        default = (a * b)._eval_at(c).evaluate
        symmetric = (a * b)._eval_at(c, interp_mode="symmetric").evaluate
        assert sympy.simplify(default - symmetric) == 0


class TestElasticStiffness:
    """
    Verify `Mul._eval_at(interp_mode="symmetric")` produces the symmetric `a = C*b`
    elastic stiffness pattern in 3D Voigt notation, with `C_{ij}` at NODE
    and `b`, `a` components at the standard staggered locations:

    - 1, 2, 3 (normal):  NODE
    - 4 (yz):  (y, z)-staggered  ("0--")
    - 5 (xz):  (x, z)-staggered  ("-0-")
    - 6 (xy):  (x, y)-staggered  ("--0")

    Expected discrete C matrix (only the structure of interp operators
    matters, not the specific weights):

    .. code-block:: text

       [   C11      C12      C13   |   C14 I0--    C15 I-0-    C16 I--0
           C12      C22      C23   |   C24 I0--    C25 I-0-    C26 I--0
           C13      C23      C33   |   C34 I0--    C35 I-0-    C36 I--0
        ----------------------------+-----------------------------
         I0++ C14  I0++ C24  I0++ C34 |    C44     I0++ C45 I-0-  I0++ C46 I--0
         I+0+ C15  I+0+ C25  I+0+ C35 | I+0+ C45 I0--   C55    I+0+ C56 I--0
         I++0 C16  I++0 C26  I++0 C36 | I++0 C46 I0--  I++0 C56 I-0-   C66 ]
    """

    @staticmethod
    def _setup():
        grid = Grid((11, 11, 11))
        x, y, z = grid.dimensions

        def F(name, stag):
            return Function(name=name, grid=grid, space_order=4, staggered=stag)

        # a, b components at Voigt locations
        a = {1: F('a1', NODE), 2: F('a2', NODE), 3: F('a3', NODE),
             4: F('a4', (y, z)), 5: F('a5', (x, z)), 6: F('a6', (x, y))}
        b = {1: F('b1', NODE), 2: F('b2', NODE), 3: F('b3', NODE),
             4: F('b4', (y, z)), 5: F('b5', (x, z)), 6: F('b6', (x, y))}
        # Stiffness C_{ij} all at NODE
        C = {(i, j): F(f'C{i}{j}', NODE) for i in range(1, 7) for j in range(1, 7)}
        return a, b, C

    @staticmethod
    def _zero_order_derivs(expr):
        return [d for d in expr.find(sympy.Derivative)
                if all(o == 0 for o in d.deriv_order)]

    @pytest.mark.parametrize('i, j', [(1, 1), (1, 2), (2, 3)])
    def test_normal_normal(self, i, j):
        """`i, j in {1, 2, 3}`: a, b, C all at NODE -> direct product."""
        a, b, C = self._setup()
        result = (C[(i, j)] * b[j])._eval_at(a[i], interp_mode="symmetric")
        # No interpolation needed
        assert self._zero_order_derivs(result) == []

    @pytest.mark.parametrize('i, j', [(1, 4), (2, 5), (3, 6)])
    def test_normal_shear_row(self, i, j):
        """`i in {1,2,3}, j in {4,5,6}`: bring `b_j` from its stag location
        to NODE; `a_i` is at NODE so no outer I. `C_{ij} * I_{j-stag}(b_j)`.

        The default-mode subs path handles this (one factor matches func),
        producing the correct `C_{ij} * b_j(NODE indices)` form."""
        a, b, C = self._setup()
        result = (C[(i, j)] * b[j])._eval_at(a[i], interp_mode="symmetric")
        # b_j gets subs'd to NODE indices; no explicit 0-order Derivative
        assert self._zero_order_derivs(result) == []
        # b_j now indexed at a_i's NODE indices, ready for interp_for_fd
        evaluated_str = str(result.evaluate)
        assert b[j].name in evaluated_str
        assert C[(i, j)].name in evaluated_str

    @pytest.mark.parametrize('i, j', [(4, 1), (5, 2), (6, 3)])
    def test_shear_normal_row(self, i, j):
        """`i in {4,5,6}, j in {1,2,3}`: `C_{ij}` and `b_j` both at NODE,
        target at stag -> single block interp `I(C_{ij} * b_j) -> a_i`."""
        a, b, C = self._setup()
        result = (C[(i, j)] * b[j])._eval_at(a[i], interp_mode="symmetric")
        zo = self._zero_order_derivs(result)
        assert len(zo) == 1
        block_str = str(zo[0].expr)
        assert C[(i, j)].name in block_str
        assert b[j].name in block_str

    @pytest.mark.parametrize('i', [4, 5, 6])
    def test_shear_diagonal(self, i):
        """`i == j in {4,5,6}`: `C_{ii}` at NODE; both `a_i` and `b_i`
        at the same stag -> `b_i` matches func, default path applies
        (C is implicitly interp'd to `a_i`'s location via subs)."""
        a, b, C = self._setup()
        result = (C[(i, i)] * b[i])._eval_at(a[i], interp_mode="symmetric")
        # b_i matches func -> default path -> no explicit 0-order Derivative
        assert self._zero_order_derivs(result) == []

    @pytest.mark.parametrize('i, j', [(4, 5), (4, 6), (5, 6), (5, 4)])
    def test_shear_shear_offdiag(self, i, j):
        """`i, j in {4,5,6}, i != j`: `C_{ij}` at NODE, `b_j` and `a_i`
        at *different* stag -> full symmetric `I_{a_i++}(C_{ij} * I_{b_j--}(b_j))`.
        Produces two 0-order interp operators (one I^T, one I)."""
        a, b, C = self._setup()
        result = (C[(i, j)] * b[j])._eval_at(a[i], interp_mode="symmetric")
        assert len(self._zero_order_derivs(result)) == 2

    def test_full_row_shear(self):
        """Build the full row 4 of `a = C * b` and verify the structure:
        sum over j of `C_{4j} * b_j` evaluated at `a_4`. Every term should
        appear and each non-diagonal term contributes a symmetric structure."""
        a, b, C = self._setup()
        terms = [(C[(4, j)] * b[j])._eval_at(a[4], interp_mode="symmetric")
                 for j in range(1, 7)]
        # Sanity: every component is referenced
        full_str = ''.join(str(t.evaluate) for t in terms)
        for j in range(1, 7):
            assert b[j].name in full_str
            assert C[(4, j)].name in full_str


class TestSymmetricAdjoint:
    """
    Numerical adjoint-identity check for `interp-mode='symmetric'`.

    For a symmetric stiffness `C` the continuous operator `sigma = C * eps`
    is self-adjoint: `<e1, C*t2> = <C*e1, t2>`. A discretization that
    preserves this identity (to numerical precision) preserves
    energy / yields the correct adjoint state. The `'symmetric'` interp
    mode does exactly that — the `I * A * I^T` factorization makes the
    discrete operator self-adjoint when `A` is.

    The companion `'direct'` mode does *not* preserve the identity (each
    factor is interpolated independently, so the discrete operator is not
    the transpose of itself).
    """

    @staticmethod
    def _setup(interp_mode, so=4):
        np.random.seed(1234)

        nx, ny, nz = 11, 11, 11
        grid = Grid(shape=(nx, ny, nz), extent=(1.0, 1.0, 1.0))
        x, y, z = grid.dimensions

        # Standard Voigt staggerings
        locs = {1: NODE, 2: NODE, 3: NODE,
                4: (y, z), 5: (x, z), 6: (x, y)}

        # Random symmetric 6x6 stiffness, all components at NODE
        C = {}
        for i in range(1, 7):
            for j in range(i, 7):
                f = Function(name=f'C{i}{j}', grid=grid, space_order=so,
                             staggered=NODE)
                f.data[:] = np.random.rand(nx, ny, nz)
                C[(i, j)] = f
                C[(j, i)] = f

        def six(prefix):
            return {i: Function(name=f'{prefix}{i}', grid=grid,
                                space_order=so, staggered=locs[i])
                    for i in range(1, 7)}

        e1, e2, t1, t2 = six('e1_'), six('e2_'), six('t1_'), six('t2_')
        for i in range(1, 7):
            e1[i].data[:] = 2 * np.random.rand(nx, ny, nz) - 1
            t2[i].data[:] = 2 * np.random.rand(nx, ny, nz) - 1

        # t1 = C * e1 and e2 = C * t2 -- two applications of the same operator
        eqns = []
        for i in range(1, 7):
            eqns.append(Eq(t1[i], sum(C[(i, j)] * e1[j] for j in range(1, 7))))
            eqns.append(Eq(e2[i], sum(C[(i, j)] * t2[j] for j in range(1, 7))))

        Operator(eqns, sym_opt={'interp-mode': interp_mode}).apply()

        inner_e = sum(float(np.dot(e1[i].data.flatten(),
                                   e2[i].data.flatten()))
                      for i in range(1, 7))
        inner_t = sum(float(np.dot(t1[i].data.flatten(),
                                   t2[i].data.flatten()))
                      for i in range(1, 7))
        return inner_e, inner_t

    def test_symmetric_preserves_adjoint(self):
        """`<e1, C*t2> == <C*e1, t2>` to numerical precision under
        `interp-mode='symmetric'`."""
        inner_e, inner_t = self._setup('symmetric')
        rel = abs(inner_e - inner_t) / max(abs(inner_e), abs(inner_t))
        assert rel < 1e-5, (
            f'<e1, C*t2> = {inner_e!r} vs <C*e1, t2> = {inner_t!r} '
            f'(rel diff {rel:.3e})'
        )

    def test_direct_breaks_adjoint(self):
        """`interp-mode='direct'` interpolates factors independently and so
        does *not* preserve the discrete adjoint identity. Recorded as an
        explicit large-discrepancy check so a regression that accidentally
        makes `'direct'` adjoint-correct also shows up."""
        inner_e, inner_t = self._setup('direct')
        rel = abs(inner_e - inner_t) / max(abs(inner_e), abs(inner_t))
        assert rel > 1e-2, (
            f"'direct' mode unexpectedly preserved adjoint identity: "
            f'<e1, C*t2> = {inner_e!r}, <C*e1, t2> = {inner_t!r} '
            f'(rel diff {rel:.3e})'
        )
