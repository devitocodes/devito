from devito.types import Eq, Function, TimeFunction


def superstep_generator_iterative(field, stencil, k, nt=0):
    """
    Generate superstep iteratively:
    Aʲ⁺¹ = A·Aʲ
    """
    # New fields, for vector formulation both current and previous timestep are needed
    name = field.name
    grid = field.grid
    u = TimeFunction(name=f'{name}_ss', grid=grid, time_order=2, space_order=2*k)
    u_prev = TimeFunction(name=f'{name}_ss_p', grid=grid, time_order=2, space_order=2*k)

    superstep_solution_transfer(field, u, u_prev, nt)

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


def superstep_generator(field, stencil, k, nt=0):
    """
    Generate superstep using a binary decomposition:
    A^k = aⱼ A^2ʲ × ... × a₂ A^2² × a₁ A² × a₀ A
    where k = aⱼ·2ʲ + ... + a₂·2² + a₁·2¹ + a₀·2⁰
    """
    # New fields, for vector formulation both current and previous timestep are needed
    name = field.name
    grid = field.grid
    # time_order of `field` needs to be 2
    if field.time_order != 2:
        raise ValueError(
            'Superstepping is currently only supports `time_order=2`'
        )
    u = TimeFunction(
        name=f'{name}_ss',
        grid=grid,
        time_order=field.time_order,
        space_order=2*k
    )
    u_prev = TimeFunction(
        name=f'{name}_ss_p',
        grid=grid,
        time_order=field.time_order,
        space_order=2*k
    )

    superstep_solution_transfer(field, u, u_prev, nt)

    # Substitute new fields into stencil
    ss_stencil = stencil.subs({field: u, field.backward: u_prev}, postprocess=False)
    ss_stencil = ss_stencil.expand().expand(add=True, nest=True)

    # Binary decomposition algorithm (see docstring):
    # Calculate the binary decomposition of the exponent (k) and accumulate the
    # resultant operator
    current = (ss_stencil, u)
    q, r = divmod(k, 2)
    accumulate = current if r else (1, 1)
    while q:
        q, r = divmod(q, 2)
        current = _combine_superstep(current, current, u, u_prev, k)
        if r:
            accumulate = _combine_superstep(accumulate, current, u, u_prev, k)

    return u, u_prev, Eq(u.forward, accumulate[0]), Eq(u_prev.forward, accumulate[1])


def superstep_solution_transfer(old, new, new_p, nt):
    """
    Transfer state from a previous TimeFunction to a 2 field superstep
    Used after injecting source using standard timestepping.
    """
    # This method is completely generic for future development, but currently
    # only time_order == 2 is implemented!
    idx = nt % (old.time_order + 1) if old.save is None else -1
    for ii in range(old.time_order):
        new.data[ii, :] = old.data[idx - ii - 1]
        new_p.data[ii, :] = old.data[idx - ii - 2]


def _combine_superstep(stencil_a, stencil_b, u, u_prev, k):
    """
    Combine two arbitrary order supersteps
    """
    # Placeholder fields for forming the superstep
    grid = u.grid
    # Can I use a TempFunction here?
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
