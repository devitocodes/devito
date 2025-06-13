import numpy as np
import sympy
from sympy import Rational, Matrix

import pytest

from devito import VectorFunction, TensorFunction, VectorTimeFunction, TensorTimeFunction
from devito import (
    Grid, Function, TimeFunction, Dimension, Eq, div, grad, curl, laplace, diag
)
from devito.symbolics import retrieve_derivatives
from devito.types import NODE


def dimify(dimensions):
    assert isinstance(dimensions, str)
    return tuple(Dimension(name=i) for i in dimensions.split())


@pytest.mark.parametrize('func_type, ndim', [
    (TensorFunction, 2), (TensorFunction, 3),
    (TensorTimeFunction, 2), (TensorTimeFunction, 3)])
def test_tensor_staggering(func_type, ndim):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid)

    dims = grid.dimensions
    assert np.all(f[i, i].staggered == NODE for i in range(ndim))
    assert np.all(f[i, j].staggered == (dims[i], dims[j])
                  for i in range(ndim) for j in range(ndim) if i != j)


@pytest.mark.parametrize('func_type, ndim, sym', [
    (TensorFunction, 2, 'noop'), (TensorFunction, 3, 'noop'),
    (TensorFunction, 2, 'diag'), (TensorFunction, 3, 'diag'),
    (TensorFunction, 2, 'symm'), (TensorFunction, 3, 'symm'),
    (TensorTimeFunction, 2, 'noop'), (TensorTimeFunction, 3, 'noop'),
    (TensorTimeFunction, 2, 'diag'), (TensorTimeFunction, 3, 'diag'),
    (TensorTimeFunction, 2, 'symm'), (TensorTimeFunction, 3, 'symm')])
def test_tensor_symmetries(func_type, ndim, sym):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid, symmetric=(sym == 'symm'),
                  diagonal=(sym == 'diag'))
    if sym == 'symm':
        assert np.all(f[i, j] == f[j, i] for i in range(ndim) for j in range(ndim))
    if sym == 'diag':
        assert np.all(f[i, j] == 0 for i in range(ndim) for j in range(ndim) if i != j)
    if sym == 'noop':
        assert np.all(f[i, j] != f[j, i] for i in range(ndim) for j in range(ndim))
        assert np.all(f[i, j] != 0 for i in range(ndim) for j in range(ndim) if i != j)


@pytest.mark.parametrize('func_type, ndim', [
    (VectorFunction, 2), (VectorFunction, 3),
    (VectorTimeFunction, 2), (VectorTimeFunction, 3)])
def test_vector_staggering(func_type, ndim):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid)
    dims = grid.dimensions
    assert np.all(f[i].staggered == dims[i] for i in range(ndim))


@pytest.mark.parametrize('func_type, ndim', [
    (TensorFunction, 2), (TensorFunction, 3),
    (TensorTimeFunction, 2), (TensorTimeFunction, 3),
    (VectorFunction, 2), (VectorFunction, 3),
    (VectorTimeFunction, 2), (VectorTimeFunction, 3)])
def test_tensor_grid(func_type, ndim):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid)
    assert np.all(ff.grid == grid for ff in f)


@pytest.mark.parametrize('func_type, ndim', [
    (TensorFunction, 2), (TensorFunction, 3),
    (TensorTimeFunction, 2), (TensorTimeFunction, 3),
    (VectorFunction, 2), (VectorFunction, 3),
    (VectorTimeFunction, 2), (VectorTimeFunction, 3)])
def test_tensor_space_order(func_type, ndim):
    grid = Grid(tuple([5]*ndim))
    f = func_type(name="f", grid=grid, space_order=10)
    assert np.all(ff.space_order == 10 for ff in f)


@pytest.mark.parametrize('func1, func2, out_type', [
    (Function, VectorFunction, VectorFunction),
    (Function, VectorTimeFunction, VectorTimeFunction),
    (TimeFunction, VectorTimeFunction, VectorTimeFunction),
    (Function, TensorFunction, TensorFunction),
    (Function, TensorTimeFunction, TensorTimeFunction),
    (TimeFunction, TensorTimeFunction, TensorTimeFunction),
    (TensorFunction, VectorFunction, VectorFunction),
    (TensorFunction, VectorTimeFunction, VectorTimeFunction),
    (TensorTimeFunction, VectorFunction, VectorTimeFunction),
    (TensorTimeFunction, VectorTimeFunction, VectorTimeFunction),
    (TensorTimeFunction, TensorFunction, TensorTimeFunction),
    (TensorTimeFunction, VectorTimeFunction, TensorTimeFunction)])
def test_tensor_matmul(func1, func2, out_type):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = func2(name="f2", grid=grid)
    assert isinstance(f1*f2, out_type)


@pytest.mark.parametrize('func1, func2, out_type', [
    (VectorFunction, TensorFunction, TensorFunction),
    (VectorTimeFunction, TensorFunction, TensorTimeFunction),
    (VectorFunction, TensorTimeFunction, TensorTimeFunction),
    (VectorTimeFunction, TensorTimeFunction, TensorTimeFunction)])
def test_tensor_matmul_T(func1, func2, out_type):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = func2(name="f2", grid=grid)
    assert isinstance(f1.T*f2, out_type)


@pytest.mark.parametrize('func1, func2, out_type', [
    (VectorFunction, VectorFunction, TensorFunction),
    (VectorTimeFunction, VectorTimeFunction, TensorTimeFunction),
    (VectorFunction, VectorTimeFunction, TensorTimeFunction)])
def test_tensor_outer(func1, func2, out_type):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = func2(name="f2", grid=grid)
    assert isinstance(f1*f2.T, out_type)


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction,
                                   VectorFunction, VectorTimeFunction])
def test_tensor_custom_dims(func1):
    dimensions = dimify('i j k')
    ndim = 3
    f = func1(name="f", dimensions=dimensions, shape=(2, 3, 4))
    assert np.all(f[i, i].staggered == NODE for i in range(ndim))
    assert np.all(f[i, j].staggered == (dimensions[i], dimensions[j])
                  for i in range(3) for j in range(3) if i != j)


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction])
def test_tensor_transpose(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid, symmetric=False)
    f2 = f1.T
    assert np.all([f1[i, j] == f2[j, i] for i in range(3) for j in range(3)])


@pytest.mark.parametrize('func1', [VectorFunction, VectorTimeFunction])
def test_vector_transpose(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = f1.T
    assert f2.shape == f1.shape[::-1]
    assert np.all([f1[i] == f2[i] for i in range(3)])


@pytest.mark.parametrize('func1', [VectorFunction, VectorTimeFunction])
def test_vector_transpose_deriv(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = f1.dx.T
    assert all([f2[i] == f1[i].dx.T for i in range(3)])


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction])
def test_tensor_transpose_deriv(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = f1.dx.T
    assert np.all([f2[i, j] == f1[j, i].dx.T for i in range(3) for j in range(3)])


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction,
                                   VectorFunction, VectorTimeFunction])
def test_transpose_vs_T(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = f1.dx.T
    f3 = f1.dx.transpose(inner=True)
    f4 = f1.dx.transpose(inner=False)
    # inner=True is the same as T
    assert f3 == f2
    # inner=False doesn't tranpose inner derivatives
    for f4i, f2i in zip(f4, f2):
        assert f4i == f2i.T


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction,
                                   VectorFunction, VectorTimeFunction])
def test_tensor_fd(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    assert np.all([f.dx == f2 for f, f2 in zip(f1, f1.dx)])


@pytest.mark.parametrize('func1, symm, diag, expected',
                         [(TensorFunction, False, False, 9),
                          (TensorFunction, True, False, 6),
                          (TensorFunction, False, True, 3),
                          (TensorTimeFunction, False, False, 9),
                          (TensorTimeFunction, True, False, 6),
                          (TensorTimeFunction, False, True, 3)])
def test_tensor_eq(func1, symm, diag, expected):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid, symmetric=symm, diagonal=diag)
    for attr in f1[0]._fd:
        # Skip rotated derivatives as the staggering isn't compatible with it
        if '45' in attr:
            continue
        eq = Eq(f1, getattr(f1, attr))
        assert len(eq.evaluate._flatten) == expected


@pytest.mark.parametrize('func1', [VectorTimeFunction, TensorTimeFunction])
def test_save(func1):
    grid = Grid(tuple([5]*3))
    time = grid.time_dim
    f1 = func1(name="f1", grid=grid, save=10, time_order=1)
    assert all(ff.indices[0] == time for ff in f1)
    assert all(ff.indices[0] == time + time.spacing for ff in f1.forward)
    assert all(ff.indices[0] == time + 2*time.spacing for ff in f1.forward.forward)
    assert all(ff.indices[0] == time - time.spacing for ff in f1.backward)
    assert all(ff.shape[0] == 10 for ff in f1)


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction])
def test_sympy_matrix(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)

    sympy_f1 = f1.as_mutable()
    vec = sympy.Matrix(3, 1, np.random.rand(3))
    mat = sympy.Matrix(3, 3, np.random.rand(3, 3).ravel())
    assert all(sp - dp == 0 for sp, dp in zip(mat * f1, mat * sympy_f1))
    assert all(sp - dp == 0 for sp, dp in zip(f1 * vec, sympy_f1 * vec))


@pytest.mark.parametrize('func1', [VectorFunction, VectorTimeFunction])
def test_sympy_vector(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)

    sympy_f1 = f1.as_mutable()
    mat = sympy.Matrix(3, 3, np.random.rand(3, 3).ravel())

    assert all(sp - dp == 0 for sp, dp in zip(mat * f1, mat * sympy_f1))


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction])
def test_non_devito_tens(func1):
    grid = Grid(tuple([5]*3))
    comps = sympy.Matrix(3, 3, [1, 2, 3, 2, 3, 6, 3, 6, 9])

    f1 = func1(name="f1", grid=grid, components=comps)
    f2 = func1(name="f2", grid=grid)

    assert f1.T == f1
    assert isinstance(f1.T, sympy.ImmutableDenseMatrix)
    # No devito object in the matrix components, should return a pure sympy Matrix
    assert ~isinstance(f1.T, func1)
    # Can still multiply
    f3 = f2*f1.T
    assert isinstance(f3, func1)

    for i in range(3):
        for j in range(3):
            assert f3[i, j] == sum(f2[i, k] * f1[j, k] for k in range(3))


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction])
def test_partial_devito_tens(func1):
    grid = Grid(tuple([5]*3))
    f2 = func1(name="f2", grid=grid)

    comps = sympy.Matrix(3, 3, [1, 2, f2[0, 0], 2, 3, 6, f2[0, 0], 6, 9])

    f1 = func1(name="f1", grid=grid, components=comps)

    assert f1.T == f1
    assert isinstance(f1.T, func1)
    # Should have original grid
    assert f1[0, 2].grid == grid
    # Can still multiply
    f3 = f2*f1.T
    assert isinstance(f3, func1)

    for i in range(3):
        for j in range(3):
            assert f3[i, j] == sum(f2[i, k] * f1[j, k] for k in range(3))


@pytest.mark.parametrize('shift, ndim', [(None, 2), (.5, 2), (.5, 3),
                                         (tuple([tuple([.5]*3)]*3), 3)])
def test_shifted_grad_of_vector(shift, ndim):
    grid = Grid(tuple([11]*ndim))
    f = VectorFunction(name="f", grid=grid, space_order=4)
    for order in [None, 2]:
        gf = grad(f, shift=shift, order=order).evaluate

        ref = []
        for i in range(len(grid.dimensions)):
            for j, d in enumerate(grid.dimensions):
                x0 = (None if shift is None else d + shift[i][j] * d.spacing if
                      type(shift) is tuple else d + shift * d.spacing)
                ge = getattr(f[i], 'd%s' % d.name)(x0=x0, fd_order=order)
                ref.append(ge.evaluate)

        for i, d in enumerate(gf):
            assert d == ref[i]


@pytest.mark.parametrize('shift, ndim', [(None, 2), (.5, 2), (.5, 3), ((.5, .5, .5), 3)])
def test_shifted_div_of_vector(shift, ndim):
    grid = Grid(tuple([11]*ndim))
    v = VectorFunction(name="f", grid=grid, space_order=4)
    for order in [None, 2]:
        df = div(v, shift=shift, order=order).evaluate
        ref = 0
        for i, d in enumerate(grid.dimensions):
            x0 = (None if shift is None else d + shift[i] * d.spacing if
                  type(shift) is tuple else d + shift * d.spacing)
            ref += getattr(v[i], 'd%s' % d.name)(x0=x0, fd_order=order)

        assert df == ref.evaluate


@pytest.mark.parametrize('shift, ndim', [(None, 2), (.5, 2), (.5, 3),
                                         (tuple([tuple([.5]*3)]*3), 3)])
def test_shifted_div_of_tensor(shift, ndim):
    grid = Grid(tuple([11]*ndim))
    f = TensorFunction(name="f", grid=grid, space_order=4)
    for order in [None, 2]:
        df = div(f, shift=shift, order=order).evaluate

        ref = []
        for i, a in enumerate(grid.dimensions):
            elems = []
            for j, d in reversed(list(enumerate(grid.dimensions))):
                x0 = (None if shift is None else d + shift[i][j] * d.spacing if
                      type(shift) is tuple else d + shift * d.spacing)
                ge = getattr(f[i, j], 'd%s' % d.name)(x0=x0, fd_order=order)
                elems.append(ge.evaluate)
            ref.append(sum(elems))

        for i, d in enumerate(df):
            assert d == ref[i]


@pytest.mark.parametrize('shift, ndim', [(None, 3), (.5, 3),
                                         (tuple([tuple([.5]*3)]*3), 3)])
def test_shifted_curl_of_vector(shift, ndim):
    grid = Grid(tuple([11]*ndim))
    f = VectorFunction(name="f", grid=grid, space_order=4)
    for order in [None, 2]:
        df = curl(f, shift=shift, order=order)
        drvs = retrieve_derivatives(df)
        dorder = order or 4
        for drv in drvs:
            assert drv.expr in f
            assert drv.fd_order == (dorder,)
            if shift is None:
                assert drv.x0 == {}
            else:
                assert drv.dims[0] in drv.x0


@pytest.mark.parametrize('shift, ndim', [(None, 2), (.5, 2), (.5, 3), ((.5, .5, .5), 3)])
def test_shifted_lap_of_vector(shift, ndim):
    grid = Grid(tuple([11]*ndim))
    v = VectorFunction(name="f", grid=grid, space_order=4)
    assert v.laplacian() == v.laplace
    for order in [None, 2]:
        df = v.laplacian(shift=shift, order=order)
        for (vi, dfvi) in zip(v, df):
            ref = vi.laplacian(shift=shift, order=order)
            assert dfvi == ref


@pytest.mark.parametrize('shift, ndim', [(None, 2), (Rational(1/2), 2),
                                         (Rational(1/2), 3),
                                         (tuple([tuple([.5]*3)]*3), 3)])
def test_shifted_lap_of_tensor(shift, ndim):
    grid = Grid(tuple([11]*ndim))
    v = TensorFunction(name="f", grid=grid, space_order=4)
    assert v.laplacian() == v.laplace
    for order in [None, 2]:
        df = v.laplacian(shift=shift, order=order)
        for j in range(ndim):
            ref = 0
            for i, d in enumerate(v.space_dimensions):
                x0 = (None if shift is None else d + shift[i][j] * d.spacing if
                      type(shift) is tuple else d + shift * d.spacing)
                ref += getattr(v[j, i], 'd%s2' % d.name)(x0=x0, fd_order=order)
            assert df[j] == ref


def test_basic_arithmetic():
    grid = Grid(tuple([5]*3))
    tau = TensorFunction(name="tau", grid=grid)

    # Scalar operations
    t1 = tau + 1
    assert all(t1i == ti + 1 for (t1i, ti) in zip(t1, tau))

    t1 = tau * 2
    assert all(t1i == ti * 2 for (t1i, ti) in zip(t1, tau))


def test_custom_coeffs_vector():
    grid = Grid(tuple([5]*3))
    v = VectorFunction(name="v", grid=grid, space_order=2)

    # Custom coefficients
    c = [10, 10, 10]

    dv = div(v, weights=c)
    assert dv == v[0].dx(w=c) + v[1].dy(w=c) + v[2].dz(w=c)
    assert list(dv.args[0].weights) == c

    for func in [div, grad, curl, laplace]:
        dv = func(v, weights=c)
        derivs = retrieve_derivatives(dv)
        for drv in derivs:
            assert list(drv.weights) == c


def test_custom_coeffs_tensor():
    grid = Grid(tuple([5]*3))
    tau = TensorFunction(name="tau", grid=grid, space_order=2)

    # Custom coefficients
    c = [10, 10, 10]

    dtau = div(tau, weights=c)
    for i, d in enumerate(grid.dimensions):
        assert dtau[i] == tau[i, 0].dx(w=c) + tau[i, 1].dy(w=c) + tau[i, 2].dz(w=c)
        assert list(dtau[i].args[0].weights) == c

    for func in [div, laplace]:
        dtau = func(tau, weights=c)
        derivs = retrieve_derivatives(dtau)
        for drv in derivs:
            assert list(drv.weights) == c


@pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction,
                                   VectorFunction, VectorTimeFunction])
def test_rebuild(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)
    f2 = f1.func(name="f2")
    assert f1.grid == f2.grid
    assert f2.name == 'f2'

    for (i, j) in zip(f1.flat(), f2.flat()):
        assert j.name == i.name.replace('f1', 'f2')
        assert j.grid == i.grid
        assert j.dimensions == i.dimensions

    new_dims = [Dimension(name=f'{i.name}1') for i in grid.dimensions]
    if f1.is_TimeDependent:
        new_dims = [f1[0].time_dim] + new_dims
    f3 = f1.func(dimensions=new_dims)
    assert f3.grid == grid
    assert f3.name == f1.name

    for (i, j) in zip(f1.flat(), f3.flat()):
        assert j.name == i.name
        assert j.grid == i.grid
        assert j.dimensions == tuple(new_dims)


@pytest.mark.parametrize('func1', [Function, TimeFunction,
                                   TensorFunction, TensorTimeFunction,
                                   VectorFunction, VectorTimeFunction])
def test_diag(func1):
    grid = Grid(tuple([5]*3))
    f1 = func1(name="f1", grid=grid)

    f2 = diag(f1)
    assert isinstance(f2, TensorFunction)
    if f1.is_TimeDependent:
        assert f2.is_TimeDependent
    print(f2)
    assert f2.shape == (3, 3)
    # Vector input
    if isinstance(f1, VectorFunction):
        assert all(f2[i, i] == f1[i] for i in range(3))
        assert all(f2[i, j] == 0 for i in range(3) for j in range(3) if i != j)
    # Tensor input
    elif isinstance(f1, TensorFunction):
        assert all(f2[i, i] == f1[i, i] for i in range(3))
        assert all(f2[i, j] == 0 for i in range(3) for j in range(3) if i != j)
    # Function input
    else:
        assert all(f2[i, j] == 0 for i in range(3) for j in range(3) if i != j)
        assert all(f2[i, i] == f1 for i in range(3))


@pytest.mark.parametrize('func1', [TensorFunction, VectorFunction])
def test_kwargs(func1):
    orders = Matrix([[1, 2], [3, 4]]) if func1 is TensorFunction else Matrix([1, 2])
    f = func1(name="f", grid=Grid((5, 5)), space_order=orders, symmetric=False)
    assert f.space_order == orders
