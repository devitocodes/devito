import click

from devito import Inc, Operator, Function, dimensions, info
from devito.tools import as_tuple

__all__ = ['mat_vec', 'transpose_mat_vec', 'mat_mat', 'mat_mat_sum',
           'chain_contractions']


@click.group(chain=True)
def linalg():
    """
    A set of kernels performing basic (BLAS-like) linear algebra operations.

    Upper-case letters ``A, B, C, ...`` are for matrices; lower-case letters
    ``x, y, ...`` are for vectors.
    """
    pass


def option_basic(f):
    def callback_shape(ctx, param, value):
        return as_tuple(value)

    def callback_opts(ctx, param, value):
        if value is True:
            return ('advanced', {'blockinner': True, 'blockrelax': True})
        else:
            return 'noop'

    options = [
        click.option('-ms', '--mat-shape', default=(4, 4), nargs=2, help='Matrix shape'),
        click.option('-vs', '--vec-shape', default=4, help='Vector shape',
                     callback=callback_shape),
        click.option('-o', '--optimize', default=False, is_flag=True,
                     help='Generate optimized code', callback=callback_opts)
    ]
    for option in reversed(options):
        f = option(f)
    return f


@linalg.command(name='mat-vec')
@option_basic
def cli_mat_vec(mat_shape, vec_shape, optimize, **kwargs):
    """``Ax = b``."""
    i, j = dimensions('i j')
    A = Function(name='A', shape=mat_shape, dimensions=(i, j))
    x = Function(name='x', shape=vec_shape, dimensions=(j,))
    b = Function(name='b', shape=vec_shape, dimensions=(i,))
    mat_vec(A, x, b, optimize)


@linalg.command(name='transpose-mat-vec')
@option_basic
def cli_transpose_mat_vec(mat_shape, vec_shape, optimize, **kwargs):
    """``A -> A^T, A^Tx = b``."""
    i, j = dimensions('i j')
    A = Function(name='A', shape=mat_shape, dimensions=(i, j))
    x = Function(name='x', shape=vec_shape, dimensions=(j,))
    b = Function(name='b', shape=vec_shape, dimensions=(i,))
    transpose_mat_vec(A, x, b, optimize)


@linalg.command(name='mat-mat')
@option_basic
def cli_mat_mat(mat_shape, optimize, **kwargs):
    """``AB = C``."""
    i, j, k = dimensions('i j k')
    A = Function(name='A', shape=mat_shape, dimensions=(i, j))
    B = Function(name='B', shape=mat_shape, dimensions=(j, k))
    C = Function(name='C', shape=mat_shape, dimensions=(i, k))
    mat_mat(A, B, C, optimize)


@linalg.command(name='mat-mat-sum')
@option_basic
def cli_mat_mat_sum(mat_shape, optimize, **kwargs):
    """``AB + AC = D``."""
    i, j, k = dimensions('i j k')
    A = Function(name='A', shape=mat_shape, dimensions=(i, j))
    B = Function(name='B', shape=mat_shape, dimensions=(j, k))
    C = Function(name='C', shape=mat_shape, dimensions=(j, k))
    D = Function(name='D', shape=mat_shape, dimensions=(i, k))
    mat_mat_sum(A, B, C, D, optimize)


@linalg.command(name='chain-contractions')
@option_basic
def cli_chain_contractions(mat_shape, optimize, **kwargs):
    """``AB + AC = D, DE = F``."""
    i, j, k, l = dimensions('i j k l')
    A = Function(name='A', shape=mat_shape, dimensions=(i, j))
    B = Function(name='B', shape=mat_shape, dimensions=(j, k))
    C = Function(name='C', shape=mat_shape, dimensions=(j, k))
    D = Function(name='D', shape=mat_shape, dimensions=(i, k))
    E = Function(name='E', shape=mat_shape, dimensions=(k, l))
    F = Function(name='F', shape=mat_shape, dimensions=(i, l))
    chain_contractions(A, B, C, D, E, F, optimize)


def mat_vec(A, x, b, optimize):
    """``Ax = b``."""
    op = Operator(Inc(b, A*x), opt=optimize)
    op.apply()
    info('Executed `Ax = b`')


def transpose_mat_vec(A, x, b, optimize):
    """``A -> A^T, A^Tx = b``."""
    i, j = A.indices
    op = Operator([Inc(b, A[j, i]*x)], opt=optimize)
    op.apply()
    info('Executed `A^Tx = b`')


def mat_mat(A, B, C, optimize):
    """``AB = C``."""
    op = Operator(Inc(C, A*B), opt=optimize)
    op.apply()
    info('Executed `AB = C`')


def mat_mat_sum(A, B, C, D, optimize):
    """``AB + AC = D``."""
    op = Operator(Inc(D, A*B + A*C), opt=optimize)
    op.apply()
    info('Executed `AB + AC = D`')


def chain_contractions(A, B, C, D, E, F, optimize):
    """``AB + AC = D, DE = F``."""
    op = Operator([Inc(D, A*B + A*C), Inc(F, D*E)], opt=optimize)
    op.apply()
    info('Executed `AB + AC = D, DE = F`')


if __name__ == "__main__":
    linalg()
