import numpy as np

from devito import Operator, Eq, Min, Max


def compute_residual(res, dobs, dsyn):
    """
    Computes the data residual dsyn - dobs into residual
    """
    # If we run with MPI, we have to compute the residual via an operator
    # First make sure we can take the difference and that receivers are at the
    # same position
    assert np.allclose(dobs.coordinates.data[:], dsyn.coordinates.data)
    assert np.allclose(res.coordinates.data[:], dsyn.coordinates.data)
    # Create a difference operator
    diff_eq = Eq(res, dsyn - dobs)
    Operator(diff_eq)()


def update_with_box(vp, alpha, dm, vmin=2.0, vmax=3.5):
    """
    Apply gradient update in-place to vp with box constraint

    Notes:
    ------
    For more advanced algorithm, one will need to gather the non-distributed
    velocity array to apply constrains and such.
    """
    update = vp + alpha * dm
    update_eq = Eq(vp, Max(Min(update, vmax), vmin))
    Operator(update_eq)()
