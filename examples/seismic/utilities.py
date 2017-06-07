from devito.interfaces import TimeData, t, time, Forward
from devito.operator import Operator

from sympy.abc import s, h
from sympy import Eq, solve


def modelling(model, src, rec, save=False):
    # Define the wavefield with the size of the model and the time dimension
    u = TimeData(name="u", shape=model.shape_domain, time_order=2, space_order=4,
                 save=save, time_dim=src.nt)

    # We can now write the PDE
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

    # This discrete PDE can be solved in a time-marching way updating u(t+dt) from the previous time step
    # Devito as a shortcut for u(t+dt) which is u.forward. We can then rewrite the PDE as
    # a time marching updating equation known as a stencil using sympy functions

    stencil = Eq(u.forward, solve(pde, u.forward)[0])
    # Finally we define the source injection and receiver read function to generate the corresponding code
    src_term = src.inject(field=u, expr=src * model.critical_dt ** 2 / model.m,
                          u_t=u.indices[0] + 1, p_t=time, offset=model.nbpml)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u, u_t=u.indices[0], p_t=time, offset=model.nbpml)
    print(rec_term)

    op = Operator([stencil] + src_term + rec_term,
                  subs={s: model.critical_dt, h: model.spacing[0]})

    return op
