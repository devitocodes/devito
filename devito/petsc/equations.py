import numpy as np

from devito import Max, Min
from devito.petsc.types.array import PETScArray
from devito.petsc.types.dimension import SpaceDimMax, SpaceDimMin, SubDimMax, SubDimMin
from devito.petsc.types.equation import CallbackEq, ConstrainBC
from devito.petsc.types.multigrid import (
    CoarseningFactorScalar, FineGlobalStartScalar, GlobalStartScalar
)
from devito.symbolics import retrieve_dimensions, retrieve_indexed, uxreplace
from devito.types.basic import Scalar
from devito.types.dimension import CustomDimension, SpaceDimension


def lower_exprs_petsc(expressions, **kwargs):

    # Process `ConstrainBC` equations
    expressions = constrain_essential_bcs(expressions, **kwargs)

    # Remap fine-grid, non-target reads inside multilevel callbacks
    expressions = lower_multilevel_fine_grid_accesses(expressions, **kwargs)

    return expressions


def lower_multilevel_fine_grid_accesses(expressions, **kwargs):
    """
    """
    ml_exprs = [e for e in expressions
                if isinstance(e, CallbackEq) and e.is_multilevel]
    if not ml_exprs:
        return expressions

    new_exprs = []
    for e in expressions:
        if not (isinstance(e, CallbackEq) and e.is_multilevel):
            new_exprs.append(e)
            continue

        target = e.lhs.function.target
        fine_grid = target.grid
        distributor = fine_grid.distributor

        factor = CoarseningFactorScalar('factor', depth=0)

        syms = {}

        def _gsc_gsf(d):
            if d not in syms:
                root = Scalar(name=f'{d.name}_m_glb', dtype=np.int32, is_const=True)
                gsc = GlobalStartScalar(f'{d.name}_m_glb', dim=d,
                                        distributor=distributor, root=root)
                gsf = FineGlobalStartScalar(f'{d.name}_m_glb_d0', dim=d,
                                            distributor=distributor, root=root)
                syms[d] = (gsc, gsf)
            return syms[d]

        mapper = {}
        for i in retrieve_indexed(e):
            if i.function.grid is not fine_grid or isinstance(i.function, PETScArray):
                continue
            dim_map = {}
            for a in i.indices:
                for s in a.free_symbols:
                    root = getattr(s, 'root', s)
                    if root in fine_grid.dimensions:
                        gsc, gsf = _gsc_gsf(root)
                        dim_map[s] = factor * (s + gsc) - gsf
            if dim_map:
                mapper[i] = uxreplace(i, dim_map)

        new_exprs.append(uxreplace(e, mapper) if mapper else e)

    return new_exprs


def constrain_essential_bcs(expressions, **kwargs):
    """
    Expand loop bounds for `ConstrainBC` expressions so that each MPI rank
    iterates over all locally visible constrained points, including those in
    the halo. PETSc requires each rank to report all constrained nodes in its
    local data region. The loops are not used for data access — only to
    identify which local indices are constrained.
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
