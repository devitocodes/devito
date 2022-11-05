from collections import OrderedDict, namedtuple, defaultdict
from itertools import product
from operator import attrgetter

from cached_property import cached_property
from sympy import Max, Min

from devito.data import CORE, OWNED, LEFT, CENTER, RIGHT
from devito.ir.support import Forward, Scope
from devito.tools import Tag, as_tuple, filter_ordered, flatten, frozendict, is_integer
from devito.types import Grid

__all__ = ['HaloScheme', 'HaloSchemeEntry', 'HaloSchemeException']


class HaloSchemeException(Exception):
    pass


class HaloLabel(Tag):
    pass
NONE = HaloLabel('none')  # noqa
IDENTITY = HaloLabel('identity')
STENCIL = HaloLabel('stencil')


HaloSchemeEntry = namedtuple('HaloSchemeEntry', 'loc_indices loc_dirs halos')

Halo = namedtuple('Halo', 'dim side')

OMapper = namedtuple('OMapper', 'core owned')


class HaloScheme(object):

    """
    A HaloScheme describes a set of halo exchanges through a mapper:

        `M : Function -> HaloSchemeEntry`

    Where `HaloSchemeEntry` is a (named) 3-tuple:

        `(loc_indices={}, loc_dirs={}, halos=[(Dimension, DataSide), ...])`

    `loc_indices` is a dict telling how to access/insert the halo along non-halo
    indices. For example, consider the Function `u(t, x, y)`. Assume `x` and
    `y` require a halo exchange. The question is: once the halo exchange is
    performed, at what offset in `t` should it be placed? should it be at `u(0,
    ...)` or `u(1, ...)` or even `u(t-1, ...)`? `loc_indices` has as many
    entries as non-halo Dimensions, and each entry provides symbolic information
    about how to access the corresponding non-halo Dimension. For example, here
    `loc_indices` could be `{t: t-1}`.

    `loc_dirs` is a dict describing the iteration direction of each Dimension
    in `loc_indices`. This information is used to perform operations over a set
    of HaloSchemeEntries, such as computing their union.

    `halos` is a list of 2-tuples `(Dimension, DataSide)`. This is metadata
    about the halo exchanges, such as the Dimensions along which a halo exchange
    is expected to be performed.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        The expressions for which the HaloScheme is derived.
    ispace : IterationSpace
        The iteration space of the expressions.
    """

    def __init__(self, exprs, ispace):
        # Derive the halo exchanges
        self._mapper = frozendict(classify(exprs, ispace))

        # Track the IterationSpace offsets induced by SubDomains/SubDimensions.
        # These should be honored in the derivation of the `omapper`
        self._honored = {}
        # SubDimensions are not necessarily included directly in
        # ispace.dimensions and hence we need to first utilize the `_defines` method
        dims = set().union(*[d._defines for d in ispace.dimensions
                             if d._defines & self.dimensions])
        subdims = [d for d in dims if d.is_Sub and not d.local]
        for i in subdims:
            ltk, _ = i.thickness.left
            rtk, _ = i.thickness.right
            self._honored[i.root] = frozenset([(ltk, rtk)])
        self._honored = frozendict(self._honored)

    def __repr__(self):
        fnames = ",".join(i.name for i in set(self._mapper))
        return "HaloScheme<%s>" % fnames

    def __eq__(self, other):
        return isinstance(other, HaloScheme) and self.fmapper == other.fmapper

    def __len__(self):
        return len(self._mapper)

    def __hash__(self):
        return (self._mapper.__hash__(), self.honored.__hash__())

    @classmethod
    def build(cls, fmapper, honored):
        obj = object.__new__(HaloScheme)
        obj._mapper = frozendict(fmapper)
        obj._honored = frozendict(honored)
        return obj

    @classmethod
    def union(self, halo_schemes):
        """
        Create a new HaloScheme from the union of a set of HaloSchemes.
        """
        fmapper = {}
        honored = {}
        for i in as_tuple(halo_schemes):
            # Compute the `fmapper `union`
            for k, v in i.fmapper.items():
                hse = fmapper.setdefault(k, v)

                if hse.loc_indices != v.loc_indices:
                    # The `loc_dirs` must match otherwise it'd be a symptom there's
                    # something horribly broken elsewhere!
                    assert hse.loc_dirs == v.loc_dirs
                    assert list(hse.loc_indices) == list(v.loc_indices)

                    # NOTE: we rarely end up here, but still, if we do, we need
                    # to compute the union of the loc_indices as well and then
                    # take the min/max along each direction, just like when we
                    # build an HaloScheme
                    raw_loc_indices = {d: (hse.loc_indices[d], v.loc_indices[d])
                                       for d in hse.loc_indices}
                    loc_indices, loc_dirs = process_loc_indices(raw_loc_indices,
                                                                hse.loc_dirs)
                else:
                    loc_indices, loc_dirs = hse.loc_indices, hse.loc_dirs

                # Potentially more halo exchanges required
                halos = hse.halos | v.halos

                fmapper[k] = HaloSchemeEntry(loc_indices, loc_dirs, halos)

            # Compute the `honored` union
            for d, v in i.honored.items():
                honored[d] = honored.get(d, frozenset()) | v

        return HaloScheme.build(fmapper, honored)

    @property
    def honored(self):
        return self._honored

    @cached_property
    def fmapper(self):
        return OrderedDict([(i, self._mapper[i]) for i in
                            sorted(self._mapper, key=attrgetter('name'))])

    @cached_property
    def is_void(self):
        return len(self.fmapper) == 0

    @cached_property
    def omapper(self):
        """
        Logical decomposition of the DOMAIN region into OWNED and CORE sub-regions.

        This is "cumulative" over all DiscreteFunctions in the HaloScheme; it also
        takes into account IterationSpace offsets induced by SubDomains/SubDimensions.

        Examples
        --------
        Consider a HaloScheme comprising two one-dimensional Functions, ``u``
        and ``v``.  ``u``'s halo, on the LEFT and RIGHT DataSides respectively,
        is (2, 2), while ``v``'s is (4, 4). The situation is depicted below.

              ^^oo----------------oo^^     u
            ^^^^oooo------------oooo^^^^   v

        Where '^' represents a HALO point, 'o' a OWNED point, and '-' a CORE point.
        Together, the 'o' and '-' points constitute the DOMAIN region.

        In this example, the "cumulative" OWNED size is (left=4, right=4), that is
        the max on each DataSide across all Functions, namely ``u`` and ``v``.

        The ``omapper`` will contain the following entries:

            [(((d, CORE, CENTER),), {d: (d_m + 4, d_M - 4)}),
             (((d, OWNED, LEFT),), {d: (d_m, min(d_m + 3, d_M))}),
             (((d, OWNED, RIGHT),), {d: (max(d_M - 3, d_m), d_M)})]

        In presence of SubDomains (or, more generally, iteration over SubDimensions),
        the "true" DOMAIN is actually smaller. For example, consider again the
        example above, but now with a SubDomain that excludes the first ``nl``
        and the last ``nr`` DOMAIN points, where ``nl >= 0`` and ``nr >= 0``. Often,
        ``nl`` and ``nr`` are referred to as the "thickness" of the SubDimension (see
        also SubDimension.__doc__). For example, the situation could be as below

              ^^ooXXX----------XXXoo^^     u
            ^^^^ooooX----------Xoooo^^^^   v

        Where 'X' is a CORE point excluded by the computation due to the SubDomain.
        Here, the 'o' points are outside of the SubDomain, but in general they could
        also be inside. The ``omapper`` is constructed taking into account that
        SubDomains are iterated over with min point ``d_m + nl`` and max point
        ``d_M - nr``. Here, the ``omapper`` is:

            [(((d, CORE, CENTER),), {d: (d_m + 4, d_M - 4),
                                     nl: (max(nl - 4, 0),),
                                     nr: (max(nr - 4, 0),)}),
             (((d, OWNED, LEFT),), {d: (d_m, min(d_m + 3, d_M - nr)),
                                    nl: (nl,),
                                    nr: (0,)}),
             (((d, OWNED, RIGHT),), {d: (max(d_M - 3, d_m + nl), d_M),
                                     nl: (0,),
                                     nr: (nr,)})]

        To convince ourselves that this makes sense, we consider a number of cases.
        For now, we assume ``|d_M - d_m| > HALO``, that is the left-HALO and right-HALO
        regions do not overlap.

            1. The SubDomain thickness is 0, which is like there were no SubDomains.
               By instantiating the template above with ``nl = 0`` and ``nr = 0``,
               it is trivial to see that we fall back to the non-SubDomain case.

            2. The SubDomain thickness is as big as the HALO region size, that is
               ``nl = 4`` and ``nr = 4``. The ``omapper`` is such that no iterations
               will be performed in the OWNED regions (i.e., "everything is CORE").

            3. The SubDomain left-thickness is smaller than the left-HALO region size,
               while the SubDomain right-thickness is larger than the right-Halo region
               size. This means that some left-OWNED points are within the SubDomain,
               while the RIGHT-OWNED are outside. For example, take ``nl = 1`` and
               ``nr = 5``; the iteration regions will then be:

                - (CORE, CENTER): {d: (d_m + 4, d_M - 4), nl: (0,), nr: (1,)}, so
                  the min point is ``d_m + 4``, while the max point is ``d_M - 5``.

                - (OWNED, LEFT): {d: (d_m, d_m + 3), nl: (1,), nr: (0,)}, so the
                  min point is ``d_m + 1``, while the max point is ``dm + 3``.

                - (OWNED, RIGHT): {d: (d_M - 3, d_M), nl: (0,), nr: (5,)}, so the
                  min point is ``d_M - 3``, while the max point is ``d_M - 5``,
                  which implies zero iterations in this region.

        Let's now assume that the left-HALO and right-HALO regions overlap. For example,
        ``d_m = 0`` and ``d_M = 1`` (i.e., the DOMAIN only has two points), with the HALO
        size that is still (4, 4).

            4. Let's take ``nl = 1`` and ``nr = 0``. That is, only one point is in
               the SubDomain and should be updated. We again instantiate the iteration
               regions and obtain:

                - (CORE, CENTER): {d: (d_m + 4, d_M - 4), nl: (0,), nr: (0,)}, so
                  the min point is ``d_m + 4 = 4``, while the max point is
                  ``d_M - 4 = -3``, which implies zero iterations in this region.

                - (OWNED, LEFT): {d: (d_m, min(d_m + 3, d_M - nr)), nl: (1,), nr: (0,)},
                  so the min point is ``d_m + 1 = 1``, while the max point is
                  ``min(d_m + 3, d_M - nr) = min(3, 1) = 1``, which implies that there
                  is exactly one point in this region.

                - (OWNED, RIGHT): {d: (max(d_M - 3, d_m + nl), d_M), nl: (0,), nr: (0,)},
                  so the min point is ``max(d_M - 3, d_m + nl) = max(-2, 1) = 1``, while
                  the max point is ``d_M = 1``, which implies that there is exactly one
                  point in this region, and this point is redundantly computed as it's
                  logically the same as that in the (OWNED, LEFT) region.

        Notes
        -----
        For each Function, the '^' and 'o' are exactly the same on *all MPI
        ranks*, so the output of this method is guaranteed to be consistent
        across *all MPI ranks*.
        """
        items = [((d, CENTER), (d, LEFT), (d, RIGHT)) for d in self.dimensions]

        processed = []
        for item in product(*items):
            where = []
            mapper = {}
            for d, s in item:
                osl, osr = self.owned_size[d]

                # Handle SubDomain/SubDimensions to-honor offsets
                nl = Max(0, *[i for i, _ in self.honored.get(d, [])])
                nr = Max(0, *[i for _, i in self.honored.get(d, [])])

                if s is CENTER:
                    where.append((d, CORE, s))
                    mapper[d] = (d.symbolic_min + osl,
                                 d.symbolic_max - osr)
                    if nl != 0:
                        mapper[nl] = (Max(nl - osl, 0),)
                    if nr != 0:
                        mapper[nr] = (Max(nr - osr, 0),)
                else:
                    where.append((d, OWNED, s))
                    if s is LEFT:
                        mapper[d] = (d.symbolic_min,
                                     Min(d.symbolic_min + osl - 1, d.symbolic_max - nr))
                        if nl != 0:
                            mapper[nl] = (nl,)
                            mapper[nr] = (0,)
                    else:
                        mapper[d] = (Max(d.symbolic_max - osr + 1, d.symbolic_min + nl),
                                     d.symbolic_max)
                        if nr != 0:
                            mapper[nl] = (0,)
                            mapper[nr] = (nr,)
            processed.append((tuple(where), frozendict(mapper)))

        _, core = processed.pop(0)
        owned = processed

        return OMapper(core, owned)

    @cached_property
    def halos(self):
        return {f: v.halos for f, v in self.fmapper.items()}

    @cached_property
    def owned_size(self):
        mapper = {}
        for f, v in self.halos.items():
            dimensions = filter_ordered(flatten(i.dim for i in v))
            for d, s in zip(f.dimensions, f._size_owned):
                if d in dimensions:
                    maxl, maxr = mapper.get(d, (0, 0))
                    mapper[d] = (max(maxl, s.left), max(maxr, s.right))
        return mapper

    @cached_property
    def dimensions(self):
        retval = set()
        for i in set().union(*self.halos.values()):
            if isinstance(i.dim, tuple) or i.side is CENTER:
                continue
            retval.add(i.dim)
        return retval

    @cached_property
    def arguments(self):
        return self.dimensions | set(flatten(self.honored.values()))

    def project(self, functions):
        """
        Create a new HaloScheme which only retains the HaloSchemeEntries corresponding
        to the provided ``functions``.
        """
        fmapper = {f: v for f, v in self.fmapper.items() if f in as_tuple(functions)}
        return HaloScheme.build(fmapper, self.honored)

    def drop(self, functions):
        """
        Create a new HaloScheme which contains all entries in ``self`` except those
        corresponding to the provided ``functions``.
        """
        fmapper = {f: v for f, v in self.fmapper.items() if f not in as_tuple(functions)}
        return HaloScheme.build(fmapper, self.honored)


def classify(exprs, ispace):
    """
    Produce the mapper ``Function -> HaloSchemeEntry``, which describes the
    necessary halo exchanges in the given Scope.
    """
    scope = Scope(exprs)

    mapper = {}
    for f, r in scope.reads.items():
        if not f.is_DiscreteFunction:
            continue
        elif not isinstance(f.grid, Grid):
            # TODO: improve me
            continue

        # For each data access, determine if (and what type of) a halo exchange
        # is required
        halo_labels = defaultdict(set)
        for i in r:
            v = {}
            for d in i.findices:
                if f.grid.is_distributed(d):
                    if i.affine(d):
                        thl, thr = i.touched_halo(d)
                        # Note: if the left-HALO is touched (i.e., `thl = True`), then
                        # the *right-HALO* is to be sent over in a halo exchange
                        v[(d, LEFT)] = (thr and STENCIL) or IDENTITY
                        v[(d, RIGHT)] = (thl and STENCIL) or IDENTITY
                    else:
                        v[(d, LEFT)] = STENCIL
                        v[(d, RIGHT)] = STENCIL
                else:
                    v[(d, i[d])] = NONE

            # Does `i` actually require a halo exchange?
            if not any(hl is STENCIL for hl in v.values()):
                continue

            # Derive diagonal halo exchanges from the previous analysis
            combs = list(product([LEFT, CENTER, RIGHT], repeat=len(f._dist_dimensions)))
            combs.remove((CENTER,)*len(f._dist_dimensions))
            for c in combs:
                key = (f._dist_dimensions, c)
                if all(v.get((d, s)) is STENCIL or s is CENTER for d, s in zip(*key)):
                    v[key] = STENCIL

            # Finally update the `halo_labels`
            for j, hl in v.items():
                halo_labels[j].add(hl)

        if not halo_labels:
            continue

        # Distinguish between Dimensions requiring a halo exchange and those which don't
        raw_loc_indices, halos = defaultdict(list), []
        for (d, s), hl in halo_labels.items():
            try:
                hl.remove(IDENTITY)
            except KeyError:
                pass
            if not hl:
                continue
            elif len(hl) > 1:
                raise HaloSchemeException("Inconsistency found while building a halo "
                                          "scheme for `%s` along Dimension `%s`" % (f, d))
            elif hl.pop() is STENCIL:
                halos.append(Halo(d, s))
            else:
                raw_loc_indices[d].append(s)

        loc_indices, loc_dirs = process_loc_indices(raw_loc_indices,
                                                    ispace.directions)

        mapper[f] = HaloSchemeEntry(loc_indices, loc_dirs, frozenset(halos))

    return mapper


def process_loc_indices(raw_loc_indices, directions):
    """
    Process the loc_indices given in raw form.

    Consider:

        * u[t+1, x] = f(u[t, x])   => shift == 1
        * u[t-1, x] = f(u[t, x])   => shift == 1
        * u[t+1, x] = f(u[t+1, x]) => shift == 0

    Assume that `t` iterates in the Forward direction. Then:

        * In the first and second cases, the x-halo should be inserted at `t`;
        * In the third case it should be inserted at `t+1`.
    """
    loc_indices = {}
    for d, indices in raw_loc_indices.items():
        try:
            func = Max if directions[d.root] is Forward else Min
        except KeyError:
            # Max or Min is the same since `d` isn't an iteration Dimension
            func = Max

        candidates = [i for i in as_tuple(indices) if not is_integer(i)]
        m = {}
        for i in candidates:
            try:
                k = i.origin - d
            except AttributeError:
                # E.g., `i=otime`, that is a plain Symbol, or `i` not a
                # SteppingDimension
                k = i - d
            m[k] = i

        try:
            loc_indices[d] = m[func(*m.keys())]
        except KeyError:
            # E.g., `indices = [0, 1, d+1]` -- it doesn't really matter
            # what we put here, so we place 0 as it's the old behaviour
            loc_indices[d] = 0

    # Normalize directions
    known = set().union(*[i._defines for i in loc_indices])
    loc_dirs = {d: v for d, v in directions.items() if d in known}

    return frozendict(loc_indices), frozendict(loc_dirs)
