from devito.symbolics import retrieve_indexed, retrieve_dimensions, uxreplace
from devito.types.dimension import SpaceDimension, CustomDimension
from devito import Min, Max

from devito.petsc.types.equation import ConstrainBC
from devito.petsc.types.dimension import (
    SubDimMax, SubDimMin,
    SpaceDimMax, SpaceDimMin,
)


def lower_exprs_petsc(expressions, **kwargs):

    # Process `ConstrainBC` equations
    expressions = constrain_essential_bcs(expressions, **kwargs)

    return expressions


def constrain_essential_bcs(expressions, **kwargs):
    """
    """
    constrain_expressions = [e for e in expressions if isinstance(e, ConstrainBC)]
    if not constrain_expressions:
        return expressions

    sregistry = kwargs.get('sregistry')
    new_exprs = []

    # TODO: rethink
    halo_size = {e.target.function._size_halo for e in constrain_expressions}
    assert len(halo_size) == 1
    halo_size = halo_size.pop()

    all_dims = {d for e in constrain_expressions for d in extract_dims(e)}
    subdims = [d for d in all_dims if d.is_Sub and not d.local]
    space_dims = [d for d in all_dims if isinstance(d, SpaceDimension)]

    mapper = {}

    for d in subdims:
        halo = halo_size[d]

        subdim_max = SubDimMax(
            sregistry.make_name(prefix=f"{d.name}_max"), subdim=d
        )
        subdim_min = SubDimMin(
            sregistry.make_name(prefix=f"{d.name}_min"), subdim=d
        )

        mapper[d] = CustomDimension(
            name=d.name,
            symbolic_min=Max(subdim_min, d.parent.symbolic_min - halo.left),
            symbolic_max=Min(subdim_max, d.parent.symbolic_max + halo.right),
        )

    for d in space_dims:
        halo = halo_size[d]
        space_dim_max = SpaceDimMax(
            sregistry.make_name(prefix=f"{d.name}_max"), space_dim=d
        )
        space_dim_min = SpaceDimMin(
            sregistry.make_name(prefix=f"{d.name}_min"), space_dim=d
        )

        mapper[d] = CustomDimension(
            name=sregistry.make_name(prefix=f"{d.name}_expanded"),
            symbolic_min=Max(space_dim_min, d.symbolic_min - halo.left),
            symbolic_max=Min(space_dim_max, d.symbolic_max + halo.right),
        )

    # Apply mapper to expressions
    for e in expressions:
        if not isinstance(e, ConstrainBC):
            new_exprs.append(e)
            continue

        dims = extract_dims(e)
        if not dims:
            new_exprs.append(e)
            continue

        new_e = uxreplace(e, mapper)

        if e.implicit_dims:
            new_e = new_e._rebuild(
                implicit_dims=tuple(mapper.get(d, d) for d in e.implicit_dims)
            )
        new_exprs.append(new_e)
    return new_exprs


def extract_dims(expr):
    indexeds = retrieve_indexed(expr)
    dims = retrieve_dimensions(
        [i for j in indexeds for i in j.indices],
        mode="unique",
    )
    dims.update(expr.implicit_dims)
    return dims
