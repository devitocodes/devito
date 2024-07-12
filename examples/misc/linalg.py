import argparse

from devito import Inc, Operator, Function, dimensions, info
from devito.tools import as_tuple

__all__ = ['mat_vec', 'transpose_mat_vec', 'mat_mat', 'mat_mat_sum',
           'chain_contractions']


def linalg():
    """
    A set of kernels performing basic (BLAS-like) linear algebra operations.

    Upper-case letters ``A, B, C, ...`` are for matrices; lower-case letters
    ``x, y, ...`` are for vectors.
    """
    pass


def cli_mat_vec(mat_shape, vec_shape, optimize, **kwargs):
    """``Ax = b``."""
    i, j = dimensions('i j')
    A = Function(name='A', shape=mat_shape, dimensions=(i, j))
    x = Function(name='x', shape=vec_shape, dimensions=(j,))
    b = Function(name='b', shape=vec_shape, dimensions=(i,))
    mat_vec(A, x, b, optimize)


def cli_transpose_mat_vec(mat_shape, vec_shape, optimize, **kwargs):
    """``A -> A^T, A^Tx = b``."""
    i, j = dimensions('i j')
    A = Function(name='A', shape=mat_shape, dimensions=(i, j))
    x = Function(name='x', shape=vec_shape, dimensions=(j,))
    b = Function(name='b', shape=vec_shape, dimensions=(i,))
    transpose_mat_vec(A, x, b, optimize)


def cli_mat_mat(mat_shape, optimize, **kwargs):
    """``AB = C``."""
    i, j, k = dimensions('i j k')
    A = Function(name='A', shape=mat_shape, dimensions=(i, j))
    B = Function(name='B', shape=mat_shape, dimensions=(j, k))
    C = Function(name='C', shape=mat_shape, dimensions=(i, k))
    mat_mat(A, B, C, optimize)


def cli_mat_mat_sum(mat_shape, optimize, **kwargs):
    """``AB + AC = D``."""
    i, j, k = dimensions('i j k')
    A = Function(name='A', shape=mat_shape, dimensions=(i, j))
    B = Function(name='B', shape=mat_shape, dimensions=(j, k))
    C = Function(name='C', shape=mat_shape, dimensions=(j, k))
    D = Function(name='D', shape=mat_shape, dimensions=(i, k))
    mat_mat_sum(A, B, C, D, optimize)


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
    parser = argparse.ArgumentParser(description="The parent parser")

    parser.add_argument('-ms', '--mat-shape', default=(4, 4), type=int, nargs="+",
                        help='Matrix shape')
    parser.add_argument('-vs', '--vec-shape', default=4, type=int, help='Vector shape')
    parser.add_argument('-o', '--optimize', default=False, help='Generate optimized code',
                        action="store_true")

    action_parser = argparse.ArgumentParser(parents=[parser], add_help=False)

    subparsers = action_parser.add_subparsers(help='Desired problem to run',
                                              title="actions", dest='command')

    subparsers.add_parser("mat-vec", parents=[parser], add_help=False,
                          help='Ax = b')
    subparsers.add_parser('transpose-mat-vec', parents=[parser], add_help=False,
                          help='A -> A^T, A^Tx = b')
    subparsers.add_parser("mat-mat", parents=[parser], add_help=False,
                          help='AB = C')
    subparsers.add_parser('mat-mat-sum', parents=[parser], add_help=False,
                          help='AB + AC = D')
    subparsers.add_parser('chain-contractions', parents=[parser], add_help=False,
                          help='AB + AC = D')

    # Parse the arguments
    args = action_parser.parse_args()

    # Process shapes
    mat_shape = as_tuple(args.mat_shape)
    vec_shape = as_tuple(args.vec_shape)

    if not args.optimize:
        optimize = 'noop'
    else:
        optimize = ('advanced', {'blockinner': True, 'blockrelax': True})

    # Execute the corresponding function
    if args.command == 'mat-mat':
        cli_mat_mat(mat_shape=mat_shape, optimize=optimize)
    elif args.command == 'mat-vec':
        cli_mat_vec(mat_shape=mat_shape, vec_shape=vec_shape, optimize=optimize)
    elif args.command == 'transpose-mat-vec':
        cli_transpose_mat_vec(mat_shape=mat_shape, vec_shape=vec_shape, optimize=optimize)
    elif args.command == 'mat-mat-sum':
        cli_mat_mat_sum(mat_shape=mat_shape, optimize=optimize)
    elif args.command == 'chain-contractions':
        cli_chain_contractions(mat_shape=mat_shape, optimize=optimize)
