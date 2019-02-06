from collections import OrderedDict, namedtuple, defaultdict
from itertools import product
from operator import attrgetter

from cached_property import cached_property
from frozendict import frozendict

from devito.data import LEFT, CENTER, RIGHT
from devito.ir.support import Scope
from devito.tools import Tag, as_mapper, as_tuple, filter_ordered

__all__ = ['HaloScheme', 'HaloSchemeException']


class HaloSchemeException(Exception):
    pass


class HaloLabel(Tag):
    pass


NONE = HaloLabel('none')
POINTLESS = HaloLabel('pointless')
IDENTITY = HaloLabel('identity')
STENCIL = HaloLabel('stencil')


HaloSchemeEntry = namedtuple('HaloSchemeEntry', 'loc_indices halos')

Halo = namedtuple('Halo', 'dim side')


class HaloScheme(object):

    """
    A HaloScheme describes a set of halo exchanges through a mapper:

        ``M : Function -> HaloSchemeEntry``

    Where ``HaloSchemeEntry`` is a (named) 2-tuple:

        ``({loc_indices}, ((Dimension, DataSide, amount), ...))``

    The tuples (Dimension, DataSide, amount) tell the amount of data that
    a DiscreteFunction should communicate along its Dimensions.

    The dict ``loc_indices`` tells how to access/insert the halo along the
    keyed Function's non-halo indices. For example, consider the Function
    ``u(t, x, y)``. Assume ``x`` and ``y`` require a halo exchange. The
    question is: once the halo exchange is performed, at what offset in ``t``
    should it be placed? should it be at ``u(0, ...)`` or ``u(1, ...)`` or even
    ``u(t-1, ...)``? ``loc_indices`` has as many entries as non-halo
    dimensions, and each entry provides symbolic information about how to
    access the corresponding non-halo dimension. Thus, in this example
    ``loc_indices`` could be, for instance, ``{t: 0}`` or ``{t: t-1}``.

    Parameters
    ----------
    exprs : tuple of IREq
        The expressions for which the HaloScheme is built
    ispace : IterationSpace
        Description of iteration directions and sub-iterators used in ``exprs``.
    fmapper : dict, optional
        The format is the same as ``M``. When provided, ``exprs`` and ``ispace``
        are ignored. It should be used to aggregate several existing HaloSchemes
        into a single, "bigger" HaloScheme, without performing any further analysis.
    """

    def __init__(self, exprs=None, ispace=None, fmapper=None):
        if fmapper is not None:
            self._mapper = frozendict(fmapper.copy())
            return

        self._mapper = {}
        scope = Scope(exprs)
        for f, v in hs_classify(scope).items():
            halos = [Halo(*i) for i, hl in v.items() if hl is STENCIL]
            if halos:
                # There is some halo to be exchanged; *what* are the local
                # (i.e., non-halo) indices?
                dims = [i for i, hl in v.items() if hl is NONE]
                loc_indices = hs_comp_locindices(f, dims, ispace, scope)

                self._mapper[f] = HaloSchemeEntry(frozendict(loc_indices),
                                                  frozenset(halos))

        # A HaloScheme is immutable, so let's make it hashable
        self._mapper = frozendict(self._mapper)

    def __repr__(self):
        fnames = ",".join(i.name for i in set(self._mapper))
        return "HaloScheme<%s>" % fnames

    def __eq__(self, other):
        return isinstance(other, HaloScheme) and self.fmapper == other.fmapper

    def __len__(self):
        return len(self._mapper)

    def __hash__(self):
        return self._mapper.__hash__()

    @cached_property
    def fmapper(self):
        return OrderedDict([(i, self._mapper[i]) for i in
                            sorted(self._mapper, key=attrgetter('name'))])

    @cached_property
    def halos(self):
        return {f: v.halos for f, v in self.fmapper.items()}

    def union(self, others):
        """
        Create a new HaloScheme representing the union of ``self`` with other HaloSchemes.
        """
        fmapper = dict(self.fmapper)
        for i in as_tuple(others):
            for k, v in i.fmapper.items():
                hse = fmapper.setdefault(k, v)
                # At this point, the `loc_indices` must match
                if hse.loc_indices != v.loc_indices:
                    raise ValueError("Cannot compute the union of one or more HaloScheme "
                                     "when the `loc_indices` differ")
                fmapper[k] = HaloSchemeEntry(hse.loc_indices, hse.halos | v.halos)

        return HaloScheme(fmapper=fmapper)

    def project(self, functions):
        """
        Create a new HaloScheme which only retains the HaloSchemeEntries corresponding
        to the provided ``functions``.
        """
        fmapper = {k: v for k, v in self.fmapper.items() if k in as_tuple(functions)}
        return HaloScheme(fmapper=fmapper)

    def drop(self, functions):
        """
        Create a new HaloScheme which contains all entries in ``self`` except those
        corresponding to the provided ``functions``.
        """
        fmapper = {k: v for k, v in self.fmapper.items() if k not in as_tuple(functions)}
        return HaloScheme(fmapper=fmapper)


def hs_classify(scope):
    """
    A mapper ``Function -> (Dimension -> [HaloLabel]`` describing what type of
    halo exchange is expected by the DiscreteFunctions in a given Scope.
    """
    mapper = {}
    for f, r in scope.reads.items():
        if not f.is_DiscreteFunction:
            continue
        elif f.grid is None:
            # TODO: improve me
            continue
        # For each data access, determine if (and what type of) a halo exchange
        # is required
        halo_labels = defaultdict(list)
        for i in r:
            v = {}
            for d in i.findices:
                # Note: if `i` makes use of SubDimensions, we might end up adding useless
                # (yet harmless) halo exchanges.  This depends on the size of a
                # SubDimension; e.g., in rare circumstances, a SubDimension might span a
                # region that falls completely within a single MPI rank, thus requiring
                # no communication whatsoever. However, the SubDimension size is only
                # known at runtime (op.apply time), so unless one starts messing up with
                # the generated code (by adding explicit `if-then-else`s to dynamically
                # prevent a halo exchange), there is no escape from conservatively
                # assuming that some halo exchanges will be required
                if f.grid.is_distributed(d):
                    if i.affine(d):
                        if d in scope.d_from_access(i).cause:
                            v[d] = POINTLESS
                        else:
                            bl, br = i.touched_halo(d)
                            v[(d, LEFT)] = (bl and STENCIL) or IDENTITY
                            v[(d, RIGHT)] = (br and STENCIL) or IDENTITY
                    else:
                        v[(d, LEFT)] = STENCIL
                        v[(d, RIGHT)] = STENCIL
                else:
                    v[d] = NONE

            # Derive diagonal halo exchanges from the previous analysis
            combs = list(product([LEFT, CENTER, RIGHT], repeat=len(f._dist_dimensions)))
            combs.remove((CENTER,)*len(f._dist_dimensions))
            for c in combs:
                key = (f._dist_dimensions, c)
                if all(v.get((d, s)) is STENCIL or s is CENTER for d, s in zip(*key)):
                    v[key] = STENCIL

            # Finally update the `halo_labels`
            for j, hl in v.items():
                halo_labels[j].append(hl)

        # Sanity check and reductions
        for i, hl in list(halo_labels.items()):
            unique_hl = set(hl)
            if unique_hl == {STENCIL, IDENTITY}:
                halo_labels[i] = STENCIL
            elif POINTLESS in unique_hl:
                halo_labels[i] = POINTLESS
            elif len(unique_hl) == 1:
                halo_labels[i] = unique_hl.pop()
            else:
                raise HaloSchemeException("Inconsistency found while building a halo "
                                          "scheme for `%s` along Dimension `%s`" % (f, d))

        # Ignore unless an actual halo exchange is required
        if any(i is STENCIL for i in halo_labels.values()):
            mapper[f] = halo_labels

    return mapper


def hs_comp_locindices(f, dims, ispace, scope):
    """
    Map the Dimensions in ``dims`` to the local indices necessary
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
