from collections import OrderedDict, namedtuple
from itertools import product
from operator import attrgetter

from cached_property import cached_property
from frozendict import frozendict

from devito.ir.support import Scope
from devito.logger import warning
from devito.parameters import configuration
from devito.types import LEFT, RIGHT
from devito.tools import Tag, as_mapper

__all__ = ['HaloScheme', 'HaloSchemeException']


class HaloSchemeException(Exception):
    pass


class HaloLabel(Tag):
    pass


NONE = HaloLabel('none')
UNSUPPORTED = HaloLabel('unsupported')
IDENTITY = HaloLabel('identity')
STENCIL = HaloLabel('stencil')
FULL = HaloLabel('full')


HaloSchemeEntry = namedtuple('HaloSchemeEntry', 'loc_indices halos')

Halo = namedtuple('Halo', 'dim side amount')


class HaloScheme(object):

    """
    A HaloScheme describes a halo exchange pattern through a mapper: ::

        M : Function -> HaloSchemeEntry

    Where ``HaloSchemeEntry`` is a (named) 2-tuple: ::

        ({loc_indices}, ((Dimension, DataSide, amount), ...))

    The tuples (Dimension, DataSide, amount) tell the amount of data that
    a :class:`TensorFunction` should communicate along (a subset of) its
    :class:`Dimension`s.

    The dict ``loc_indices`` tells how to access/insert the halo along the
    keyed Function's non-halo indices. For example, consider the
    :class:`Function` ``u(t, x, y)``. Assume ``x`` and ``y`` require a halo
    exchange. The question is: once the halo exchange is performed, at what
    offset in ``t`` should it be placed? should it be at ``u(0, ...)`` or
    ``u(1, ...)`` or even ``u(t-1, ...)``? ``loc_indices`` has as many entries
    as non-halo dimensions, and each entry provides symbolic information about
    how to access the corresponding non-halo dimension. Thus, in this example
    ``loc_indices`` could be, for instance, ``{t: 0}`` or ``{t: t-1}``.

    :param exprs: The :class:`IREq`s for which the HaloScheme is built.
    :param ispace: A :class:`IterationSpace` describing the iteration
                   directions and the sub-iterators used by the ``exprs``.
    :param dspace: A :class:`DataSpace` describing the ``exprs`` data
                   access pattern.
    :param fmapper: (Optional) Alternatively, a HaloScheme can be built from a
                   set of known HaloSchemeEntry. If ``fmapper`` is provided,
                   then ``exprs``, ``ispace``, and ``dspace`` are ignored.
                   ``fmapper`` is a dictionary having same format as ``M``, the
                   HaloScheme mapper defined at the top of this docstring.
    """

    def __init__(self, exprs=None, ispace=None, dspace=None, fmapper=None):
        if fmapper is not None:
            self._mapper = frozendict(fmapper.copy())
            return

        self._mapper = {}

        scope = Scope(exprs)

        # *What* halo exchanges do we need?
        classification = hs_classify(scope)

        for f, v in classification.items():
            # *How much* halo do we have to exchange?
            halos = hs_comp_halos(f, [d for d, hl in v.items() if hl is STENCIL], dspace)
            halos.extend(hs_comp_halos(f, [d for d, hl in v.items() if hl is FULL]))

            # *What* are the local (i.e., non-halo) indices?
            loc_indices = hs_comp_locindices(f, [d for d, hl in v.items() if hl is NONE],
                                             ispace, dspace, scope)

            if halos:
                self._mapper[f] = HaloSchemeEntry(frozendict(loc_indices), tuple(halos))

        self._mapper = frozendict(self._mapper)

    def __repr__(self):
        fnames = ",".join(i.name for i in set(self._mapper))
        return "HaloScheme<%s>" % fnames

    def __eq__(self, other):
        return isinstance(other, HaloScheme) and self.fmapper == other.fmapper

    def __hash__(self):
        return self._mapper.__hash__()

    @cached_property
    def fmapper(self):
        return OrderedDict([(i, self._mapper[i]) for i in
                            sorted(self._mapper, key=attrgetter('name'))])

    @cached_property
    def mask(self):
        mapper = {}
        for f, v in self.fmapper.items():
            needed = [(i.dim, i.side) for i in v.halos]
            for i in product(f.dimensions, [LEFT, RIGHT]):
                if i[0] in v.loc_indices:
                    continue
                mapper.setdefault(f, OrderedDict())[i] = i in needed
        return mapper


def hs_classify(scope):
    """
    Return a mapper ``Function -> (Dimension -> [HaloLabel]`` describing what
    type of halo exchange is expected by the various :class:`TensorFunction`s
    in a :class:`Scope`.
    """
    mapper = {}
    for f, r in scope.reads.items():
        if not f.is_TensorFunction:
            continue
        elif f.grid is None:
            # TODO: improve me
            continue
        v = mapper.setdefault(f, {})
        for i in r:
            for d in i.findices:
                if i.affine(d):
                    if f.grid.is_distributed(d):
                        if i.touch_halo(d):
                            v.setdefault(d, []).append(STENCIL)
                        else:
                            v.setdefault(d, []).append(IDENTITY)
                    else:
                        v.setdefault(d, []).append(NONE)
                elif i.is_increment:
                    # A read used for a distributed local-reduction. Users are expected
                    # to deal with this data access pattern by themselves, for example
                    # by resorting to common techniques such as redundant computation
                    v.setdefault(d, []).append(UNSUPPORTED)
                elif i.irregular(d) and f.grid.is_distributed(d):
                    v.setdefault(d, []).append(FULL)

    # Sanity check and reductions
    for f, v in mapper.items():
        for d, hl in list(v.items()):
            unique_hl = set(hl)
            if unique_hl == {STENCIL, IDENTITY}:
                v[d] = STENCIL
            elif len(unique_hl) == 1:
                v[d] = unique_hl.pop()
            else:
                raise HaloSchemeException("Inconsistency found while building a halo "
                                          "scheme for `%s` along Dimension `%s`" % (f, d))

    # Drop functions needing no halo exchange
    mapper = {f: v for f, v in mapper.items()
              if any(i in [STENCIL, FULL] for i in v.values())}

    # Emit a summary warning
    for f, v in mapper.items():
        unsupported = [d for d, hl in v.items() if hl is UNSUPPORTED]
        if configuration['mpi'] and unsupported:
            warning("Distributed local-reductions over `%s` along "
                    "Dimensions `%s` detected." % (f, unsupported))

    return mapper


def hs_comp_halos(f, dims, dspace=None):
    """
    Return an iterable of 3-tuples ``[(Dimension, DataSide, amount), ...]``
    describing the amount of halo that should be exchange along the two sides of
    a set of :class:`Dimension`s.
    """
    halos = []
    for d in dims:
        if dspace is None:
            # We cannot do anything better than exchanging the full halo
            # in absence of more information
            lsize = f._extent_halo[d].left
            rsize = f._extent_halo[d].right
        else:
            # We can limit the amount of halo exchanged based on the stencil
            # radius, which is dictated by `dspace`
            v = dspace[f][d.root]
            lower, upper = v.limits if not v.is_Null else (0, 0)
            lsize = f._offset_domain[d].left - lower
            rsize = upper - f._offset_domain[d].right
        if lsize > 0:
            halos.append(Halo(d, LEFT, lsize))
        if rsize > 0:
            halos.append(Halo(d, RIGHT, rsize))
    return halos


def hs_comp_locindices(f, dims, ispace, dspace, scope):
    """
    Map the :class:`Dimension`s in ``dims`` to the local indices necessary
    to perform a halo exchange, as described in HaloScheme.__doc__.

    Examples
    --------
    1) u[t+1, x] = f(u[t, x])   => shift == 1
    2) u[t-1, x] = f(u[t, x])   => shift == 1
    3) u[t+1, x] = f(u[t+1, x]) => shift == 0
    In the first and second cases, the x-halo should be inserted at `t`,
    while in the last case it should be inserted at `t+1`.
    """
    loc_indices = {}
    for d in dims:
        func = max if ispace.is_forward(d.root) else min
        loc_index = func([i[d] for i in scope.getreads(f)], key=lambda i: i-d)
        if d.is_Stepping:
            subiters = ispace.sub_iterators.get(d.root, [])
            submap = as_mapper(subiters, lambda md: md.modulo)
            submap = {i.origin: i for i in submap[f._time_size]}
            try:
                loc_indices[d] = submap[loc_index]
            except KeyError:
                raise HaloSchemeException("Don't know how to build a HaloScheme as the "
                                          "stepping index `%s` is undefined" % loc_index)
        else:
            loc_indices[d] = loc_index
    return loc_indices
