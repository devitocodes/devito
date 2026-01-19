from sympy import Eq
from devito.symbolics import retrieve_indexed, retrieve_dimensions
from devito.petsc import EssentialBC
from devito.types.dimension import CustomBoundSubDimension
from devito import Min, Max


def lower_exprs_petsc(expressions, **kwargs):
    mapper = {}

    additional_exprs = []

    # build mapper
    for e in expressions:
        if not isinstance(e, EssentialBC):
            continue
        indexeds = retrieve_indexed(e)
        dims = retrieve_dimensions([i for j in indexeds for i in j.indices], mode='unique')

        dims = [d for d in dims if d.is_Sub and not d.local]

        for d in dims:
            # replace the dim with a new one that has a different symbolic_min and symbolic_max

            # obvs shouldn't be obtained from indexeds[0], but how should it be obtained?
            # USE e.lhs function -> the one that the BC is being applied to
            halo_size_left = indexeds[0].function._size_halo[d].left
            halo_size_right = indexeds[0].function._size_halo[d].right


            from devito.petsc.types.dimension import SubDimMax, SubDimMin



            # TODO: change name..

            # in theory this class shoulod just take in d
            # TODO: use unique name
            subdim_max = SubDimMax(d.name + '_max', subdim=d, thickness=d.thickness)
            subdim_min = SubDimMin(d.name + '_min', subdim=d, thickness=d.thickness)

            # from IPython import embed; embed()

            new_dim = CustomBoundSubDimension(
            name=d.name,
            parent=d.parent,
            thickness=d.thickness,
            local=d.local,
            custom_left=Max(subdim_min, d.parent.symbolic_min - halo_size_left),
            custom_right=Min(subdim_max, d.parent.symbolic_max + halo_size_right)
            )
            mapper[d] = new_dim

            # from IPython import embed; embed()

    # build new expressions
    for e in expressions:
        if not isinstance(e, EssentialBC):
            continue

        # build new expression
        new_e = e.subs(mapper)

        additional_exprs.append(new_e)

    # return expressions + additional_exprs
    return expressions + additional_exprs



