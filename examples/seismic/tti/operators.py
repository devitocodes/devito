from sympy import Eq, cos, sin

from devito import Operator, TimeData
from examples.seismic import PointSource, Receiver
from devito.finite_difference import centered, first_derivative, right, transpose
from devito.dimension import x, y, z, t, time


def ForwardOperator(model, source, receiver, time_order=2, space_order=4,
                    save=False, **kwargs):
    """
    Constructor method for the forward modelling operator in an acoustic media

    :param model: :class:`Model` object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    :param: u_ini : wavefield at the three first time step for non-zero initial condition

    Rotated laplacian based on the following assumptions:
    1 - Laplacian is rotation invariant.
    2 - We still need to implement the rotated version as
    the regular laplacian only look at the cartesian axxis
    3- We must have Hp + Hz = Laplacian

    Stencil goes as follow
    For Hp compute Hz and Hp = laplace - Hz
    For Hz compute directly Hz

    This guaranties that the diagonal of the operator will
    not be empty due to odd-even coupling. The FD operator
    is still self-adjoint for stability as both the laplacian and
    the implementation of Hp and Hz are self-adjoints
    """
    dt = model.critical_dt

    m, damp, epsilon, delta, theta, phi = (model.m, model.damp, model.epsilon,
                                           model.delta, model.theta, model.phi)

    # Create symbols for forward wavefield, source and receivers
    u = TimeData(name='u', shape=model.shape_domain, dtype=model.dtype,
                 save=save, time_dim=source.nt if save else None,
                 time_order=time_order, space_order=space_order)
    v = TimeData(name='v', shape=model.shape_domain, dtype=model.dtype,
                 save=save, time_dim=source.nt if save else None,
                 time_order=time_order, space_order=space_order)
    src = PointSource(name='src', ntime=source.nt, ndim=source.ndim,
                      npoint=source.npoint)
    rec = Receiver(name='rec', ntime=receiver.nt, ndim=receiver.ndim,
                   npoint=receiver.npoint)

    # Take half the space order for the first derivatives so that two applications
    # of the first derivative has the same order than the laplacian
    order1 = space_order / 2
    # Rotated laplacian
    ang0 = cos(theta)
    ang1 = sin(theta)
    if len(model.shape) == 3:
        ang2 = cos(phi)
        ang3 = sin(phi)
        # Hpu
        # Gz = -(ang1 * ang2 * u.dx + ang1 * ang3 * u.dy + ang0 * u.dz)
        # Call first-derivative instead to get the wanted order
        Gz = -(ang1 * ang2 * first_derivative(u, dim=x, side=centered, order=order1) +
               ang1 * ang3 * first_derivative(u, dim=y, side=centered, order=order1) +
               ang0 * first_derivative(u, dim=z, side=centered, order=order1))
        Gzz = (first_derivative(Gz * ang1 * ang2,
                                dim=x, side=centered, order=order1,
                                matvec=transpose) +
               first_derivative(Gz * ang1 * ang3,
                                dim=y, side=centered, order=order1,
                                matvec=transpose) +
               first_derivative(Gz * ang0,
                                dim=z, side=centered, order=order1,
                                matvec=transpose))
        Hp = u.laplace - Gzz
        # Hzv
        Gzr = -(ang1 * ang2 * first_derivative(v, dim=x, side=centered, order=order1) +
                ang1 * ang3 * first_derivative(v, dim=y, side=centered, order=order1) +
                ang0 * first_derivative(v, dim=z, side=centered, order=order1))
        Hzr = (first_derivative(Gzr * ang1 * ang2,
                                dim=x, side=centered, order=order1,
                                matvec=transpose) +
               first_derivative(Gzr * ang1 * ang3,
                                dim=y, side=centered, order=order1,
                                matvec=transpose) +
               first_derivative(Gzr * ang0,
                                dim=z, side=centered, order=order1,
                                matvec=transpose))

    else:
        # Gx1p = -(ang1 * u.dx + ang0 * u.dy)
        # Gz1r = -(ang0 * v.dx - ang1 * v.dy)
        # Call first-derivative instead to get the wanted order
        Gx = -(ang1 * first_derivative(u, dim=x, side=centered, order=order1) +
               ang0 * first_derivative(u, dim=y, side=centered, order=order1))
        Gxx1 = (first_derivative(Gx * ang1, dim=x,
                                 side=centered, order=order1,
                                 matvec=transpose) +
                first_derivative(Gx * ang0, dim=y,
                                 side=centered, order=order1,
                                 matvec=transpose))
        Hp = u.laplace - Gxx1
        # Hzr
        Gx = -(ang1 * first_derivative(v, dim=x, side=centered, order=order1) +
               ang0 * first_derivative(v, dim=y, side=centered, order=order1))
        Hzr = (first_derivative(Gx * ang1, dim=x,
                                side=centered, order=order1,
                                matvec=transpose) +
               first_derivative(Gx * ang0, dim=y,
                                side=centered, order=order1,
                                matvec=transpose))

    s = t.spacing
    # Stencils
    stencilp = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * u + (s * damp - 2.0 * m) *
         u.backward + 2.0 * s**2 * (epsilon * Hp + delta * Hzr))
    stencilr = 1.0 / (2.0 * m + s * damp) * \
        (4.0 * m * v + (s * damp - 2.0 * m) *
         v.backward + 2.0 * s**2 * (delta * Hp + Hzr))
    first_stencil = Eq(u.forward, stencilp)
    second_stencil = Eq(v.forward, stencilr)
    stencils = [first_stencil, second_stencil]

    # SOurce and receivers
    stencils += src.inject(field=u.forward, expr=src * dt * dt / m,
                           offset=model.nbpml)
    stencils += src.inject(field=v.forward, expr=src * dt * dt / m,
                           offset=model.nbpml)
    stencils += rec.interpolate(expr=u + v, offset=model.nbpml)
    # Add substitutions for spacing (temporal and spatial)
    subs = dict([(t.spacing, dt)] + [(time.spacing, dt)] +
                [(i.spacing, model.get_spacing()[j]) for i, j
                 in zip(u.indices[1:], range(len(model.shape)))])
    # Operator
    return Operator(stencils, subs=subs, name='ForwardTTI', **kwargs)
