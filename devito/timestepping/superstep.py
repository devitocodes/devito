from devito.types import Eq, Function, TimeFunction


def superstep_generator_iterative(field, stencil, k, tn=0):
    ''' Generate superstep iteratively:
    A^j+1 = A·A^j
    '''
    # New fields, for vector formulation both current and previous timestep are needed
    name = field.name
    grid = field.grid
    u = TimeFunction(name=f'{name}_ss', grid=grid, time_order=2, space_order=2*k)
    u_prev = TimeFunction(name=f'{name}_ss_p', grid=grid, time_order=2, space_order=2*k)

    superstep_solution_transfer(field, u, u_prev, tn)

    # Substitute new fields into stencil
    ss_stencil = stencil.subs({field: u, field.backward: u_prev}, postprocess=False)
    ss_stencil = ss_stencil.expand().expand(add=True, nest=True)
    current = ss_stencil

    # Placeholder fields for forming the superstep
    a_tmp = Function(name="a_tmp", grid=grid, space_order=2*k)
    b_tmp = Function(name="b_tmp", grid=grid, space_order=2*k)

    if k >= 2:
        for _ in range(k - 2):
            current = current.subs(
                {u: a_tmp, u_prev: b_tmp}, postprocess=False).subs(
                {a_tmp: ss_stencil, b_tmp: u}, postprocess=False
            )
            current = current.expand().expand(add=True, nest=True)
    else:
        current = u

    stencil_next = current.subs(
        {u: a_tmp, u_prev: b_tmp}, postprocess=False).subs(
        {a_tmp: ss_stencil, b_tmp: u}, postprocess=False
    )
    stencil_next = stencil_next.expand().expand(add=True, nest=True)
    return u, u_prev, Eq(u.forward, stencil_next), Eq(u_prev.forward, current)


def superstep_generator(field, stencil, k, tn=0):
    ''' Generate superstep using a binary decomposition:
    A^k = a_j A^2^j + ... + a_2 A^2^2 + a_1 A² + a_0 A
    '''
    # New fields, for vector formulation both current and previous timestep are needed
    name = field.name
    grid = field.grid
    u = TimeFunction(name=f'{name}_ss', grid=grid, time_order=2, space_order=2*k)
    u_prev = TimeFunction(name=f'{name}_ss_p', grid=grid, time_order=2, space_order=2*k)

    superstep_solution_transfer(field, u, u_prev, tn)

    # Substitute new fields into stencil
    ss_stencil = stencil.subs({field: u, field.backward: u_prev}, postprocess=False)
    ss_stencil = ss_stencil.expand().expand(add=True, nest=True)

    # Binary decomposition algorithm
    current = (ss_stencil, u)
    q, r = divmod(k, 2)
    accumulate = current if r else (1, 1)
    while q:
        q, r = divmod(q, 2)
        current = _combine_superstep(current, current, u, u_prev, k)
        if r:
            accumulate = _combine_superstep(accumulate, current, u, u_prev, k)

    return u, u_prev, Eq(u.forward, accumulate[0]), Eq(u_prev.forward, accumulate[1])


def superstep_solution_transfer(old, new, new_p, tn):
    ''' Transfer the timesteps from a previous simulation to a 2 field superstep simulation
    Used after injecting source using standard timestepping.
    '''
    idx = tn % 3  if old.save is None else -1
    new.data[0, :] = old.data[idx - 1]
    new.data[1, :] = old.data[idx]
    new_p.data[0, :] = old.data[idx - 2]
    new_p.data[1, :] = old.data[idx - 1]


def _combine_superstep(stencil_a, stencil_b, u, u_prev, k):
    ''' Combine two arbitrary order supersteps
    '''
    # Placeholder fields for forming the superstep
    grid = u.grid
    a_tmp = Function(name="a_tmp", grid=grid, space_order=2*k)
    b_tmp = Function(name="b_tmp", grid=grid, space_order=2*k)

    new = []
    if stencil_a == (1, 1):
        new = stencil_b
    else:
        for stencil in stencil_a:
            new_stencil = stencil.subs({u: a_tmp, u_prev: b_tmp}, postprocess=False)
            new_stencil = new_stencil.subs(
                {a_tmp: stencil_b[0], b_tmp: stencil_b[1]}, postprocess=False
            )
            new_stencil = new_stencil.expand().expand(add=True, nest=True)
            new.append(new_stencil)

    return new
