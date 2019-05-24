from collections import OrderedDict, namedtuple, defaultdict
from itertools import product
from operator import attrgetter

from cached_property import cached_property
from frozendict import frozendict

from devito.data import LEFT, CENTER, RIGHT
from devito.ir.support import Scope
from devito.tools import Tag, as_mapper, as_tuple, filter_ordered, flatten

__all__ = ['HaloScheme', 'HaloSchemeEntry', 'HaloSchemeException']


class HaloSchemeException(Exception):
    pass


class HaloLabel(Tag):
    pass


NONE = HaloLabel('none')
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
    def omapper(self):
        """
        Mapper describing the OWNED ('o'-mapper) region offset from the DOMAIN
        extremes, along each Dimension and DataSide.

        Examples
        --------
        Consider a HaloScheme comprising two one-dimensional Functions, ``u``
        and ``v``.  ``u``'s halo, on the LEFT and RIGHT DataSides respectively,
        is (2, 2), while ``v``'s is (4, 4). The situation is depicted below.

        .. code-block:: python

              xx**----------------**xx     u
            xxxx****------------****xxxx   v

        Where 'x' represents a HALO point, '*' a OWNED point, and '-' a CORE point.
        Together, '*' and '-' constitute the DOMAIN.

        In this example, the "cumulative" OWNED size is (4, 4), that is the max
        on each DataSide across all Functions, namely ``u`` and ``v``. Then, the
        ``omapper``, which provides *relative offsets*, not sizes, will be
        ``{d0: (4, -4)}``.

        Note that, for each Function, the 'x' and '*' are exactly the same on
        *all MPI ranks*, so the output of this method is guaranteed to be
        consistent across *all MPI ranks*.
        """
        mapper = {}
        for f, v in self.halos.items():
            dimensions = filter_ordered(flatten(i.dim for i in v))
            for d, s in zip(f.dimensions, f._size_owned):
                if d in dimensions:
                    mapper.setdefault(d, []).append(s)
        for k, v in list(mapper.items()):
            left, right = zip(*v)
            mapper[k] = (max(left), -max(right))
        return mapper

    @cached_property
    def halos(self):
        return {f: v.halos for f, v in self.fmapper.items()}

    @cached_property
    def dimensions(self):
        return filter_ordered(flatten(i.dim for i in set().union(*self.halos.values())))

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
                if f.grid.is_distributed(d):
                    if i.affine(d):
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
            try:
                submap = {i.origin: i for i in submap[f._time_size]}
                loc_indices[d] = submap[loc_index]
            except KeyError:
                raise HaloSchemeException("Don't know how to build a HaloScheme as the "
                                          "stepping index `%s` is undefined" % loc_index)
        else:
            loc_indices[d] = loc_index
    return loc_indices
