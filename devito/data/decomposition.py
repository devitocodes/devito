from collections.abc import Iterable

import numpy as np
from cached_property import cached_property

from devito.data.meta import LEFT
from devito.tools import is_integer

__all__ = ['Decomposition']


class Decomposition(tuple):

    """
    A decomposition of a discrete "global" domain into multiple, non-overlapping
    "local" subdomains.

    Parameters
    ----------
    items : iterable of int iterables
        The domain decomposition.
    local : int
        The local ("owned") subdomain (0 <= local < len(items)).

    Notes
    -----
    For indices, we adopt the following name conventions:

        * global/glb. Refers to the global domain.
        * local/loc. Refers to the local subdomain.

    Further, a local index can be

        * absolute/abs. Use the global domain numbering.
        * relative/rel. Use the local domain numbering.

    Examples
    --------
    In the following example, the domain consists of 8 indices, split over three
    subdomains. The instantiator owns the subdomain [3, 4].

    >>> d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7]], 1)
    >>> d
    Decomposition([0,2], <<[3,4]>>, [5,7])
    >>> d.loc_abs_min
    3
    >>> d.loc_abs_max
    4
    >>> d.loc_rel_min
    0
    >>> d.loc_rel_max
    1
    """

    def __new__(cls, items, local):
        if len(items) == 0:
            raise ValueError("The decomposition must contain at least one subdomain")
        if not all(isinstance(i, Iterable) for i in items):
            raise TypeError("Illegal Decomposition element type")
        if not is_integer(local) and (0 <= local < len(items)):
            raise ValueError("`local` must be an index in ``items``.")
        obj = super(Decomposition, cls).__new__(cls, [np.array(i) for i in items])
        obj._local = local
        return obj

    @property
    def local(self):
        return self._local

    @cached_property
    def glb_min(self):
        ret = min(min(i, default=np.inf) for i in self)
        return None if ret == np.inf else ret

    @cached_property
    def glb_max(self):
        ret = max(max(i, default=-np.inf) for i in self)
        return None if ret == -np.inf else ret

    @cached_property
    def loc_abs_numb(self):
        return self[self.local]

    @property
    def loc_empty(self):
        return self.loc_abs_numb.size == 0

    @cached_property
    def loc_abs_min(self):
        return min(self.loc_abs_numb, default=None)

    @cached_property
    def loc_abs_max(self):
        return max(self.loc_abs_numb, default=None)

    @cached_property
    def loc_rel_min(self):
        return 0

    @cached_property
    def loc_rel_max(self):
        return self.loc_abs_max - self.loc_abs_min

    @cached_property
    def size(self):
        return sum(i.size for i in self)

    def __eq__(self, o):
        if not isinstance(o, Decomposition):
            return False
        return self.local == o.local and len(self) == len(o) and\
            all(np.all(i == j) for i, j in zip(self, o))

    def __repr__(self):
        ret = []
        for i, v in enumerate(self):
            bounds = (min(v, default=None), max(v, default=None))
            item = '[]' if bounds == (None, None) else '[%d,%d]' % bounds
            if self.local == i:
                item = "<<%s>>" % item
            ret.append(item)
        return 'Decomposition(%s)' % ', '.join(ret)

    def __call__(self, *args, rel=True):
        """Alias for ``self.convert_index``."""
        return self.convert_index(*args, rel=rel)

    def convert_index(self, *args, rel=True):
        """
        Convert a global index into a relative (default) or absolute local index.

        Parameters
        ----------
        *args
            There are three possible cases:
            * int. Given ``I``, a global index, return the corresponding
              relative local index if ``I`` belongs to the local subdomain,
              None otherwise.
            * int, DataSide. Given ``O`` and ``S``, respectively a global
              offset and a side, return the relative local offset. This
              can be 0 if the local subdomain doesn't intersect with the
              region defined by the given global offset.
            * (int, int).  Given global ``(min, max)``, return ``(min', max')``
              representing the corresponding relative local min/max. If the
              input doesn't intersect with the local subdomain, then ``min'``
              and ``max'`` are two unspecified ints such that ``max'=min'-n``,
              with ``n > 1``.
            * slice(a, b). Like above, with ``min=a`` and ``max=b-1``.
              Return ``slice(min', max'+1)``.
        rel : bool, optional
            If False, convert into an absolute, instead of a relative, local index.

        Raises
        ------
        TypeError
            If the input doesn't adhere to any of the supported format.

        Examples
        --------
        In the following example, the domain consists of 12 indices, split over
        four subdomains [0, 3]. We pick 2 as local subdomain.

        >>> d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)
        >>> d
        Decomposition([0,2], [3,4], <<[5,7]>>, [8,11])

        A global index as single argument:

        >>> d.convert_index(5)
        0
        >>> d.convert_index(6)
        1
        >>> d.convert_index(7)
        2
        >>> d.convert_index(3)


        Retrieve relative local min/man given global min/max

        >>> d.convert_index((5, 7))
        (0, 2)
        >>> d.convert_index((5, 9))
        (0, 2)
        >>> d.convert_index((1, 3))
        (-1, -3)
        >>> d.convert_index((1, 6))
        (0, 1)
        >>> d.convert_index((None, None))
        (0, 2)

        Retrieve absolute local min/max given global min/max

        >>> d.convert_index((5, 9), rel=False)
        (5, 7)
        >>> d.convert_index((1, 6), rel=False)
        (5, 6)
        """

        base = self.loc_abs_min if rel is True else 0
        top = self.loc_abs_max

        if len(args) == 1:
            glb_idx = args[0]
            if is_integer(glb_idx):
                # convert_index(index)
                # -> Base case, empty local subdomain
                if self.loc_empty:
                    return None
                # -> Handle negative index
                if glb_idx < 0:
                    glb_idx = self.glb_max + glb_idx + 1
                # -> Do the actual conversion
                if glb_idx in self.loc_abs_numb:
                    return glb_idx - base
                elif self.glb_min <= glb_idx <= self.glb_max:
                    return None
                else:
                    # This should raise an exception when used to access a numpy.array
                    return glb_idx
            else:
                # convert_index((min, max))
                # convert_index(slice(...))
                if isinstance(glb_idx, tuple):
                    if len(glb_idx) != 2:
                        raise TypeError("Cannot convert index from `%s`" % type(glb_idx))
                    if self.loc_empty:
                        return (-1, -3)
                    glb_idx_min, glb_idx_max = glb_idx
                    retfunc = lambda a, b: (a, b)
                elif isinstance(glb_idx, slice):
                    if self.loc_empty:
                        return slice(-1, -3)
                    glb_idx_min = self.glb_min if glb_idx.start is None else glb_idx.start
                    glb_idx_max = self.glb_max if glb_idx.stop is None else glb_idx.stop-1
                    retfunc = lambda a, b: slice(a, b + 1, glb_idx.step)
                else:
                    raise TypeError("Cannot convert index from `%s`" % type(glb_idx))
                # -> Handle negative min/max
                if glb_idx_min is not None and glb_idx_min < 0:
                    glb_idx_min = self.glb_max + glb_idx_min + 1
                if glb_idx_max is not None and glb_idx_max < 0:
                    glb_idx_max = self.glb_max + glb_idx_max + 1
                # -> Do the actual conversion
                if glb_idx_min is None or glb_idx_min < self.loc_abs_min:
                    loc_min = self.loc_abs_min - base
                elif glb_idx_min > self.loc_abs_max:
                    return retfunc(-1, -3)
                else:
                    loc_min = glb_idx_min - base
                if glb_idx_max is None or glb_idx_max > self.loc_abs_max:
                    loc_max = self.loc_abs_max - base
                elif glb_idx_max < self.loc_abs_min:
                    return retfunc(-1, -3)
                else:
                    loc_max = glb_idx_max - base
                return retfunc(loc_min, loc_max)
        elif len(args) == 2:
            # convert_index(offset, side)
            if self.loc_empty:
                return 0
            rel_ofs, side = args
            if side is LEFT:
                abs_ofs = self.glb_min + rel_ofs
                size = self.loc_abs_max - base + 1
                return min(abs_ofs - base, size) if abs_ofs > base else 0
            else:
                abs_ofs = self.glb_max - rel_ofs
                size = top - self.loc_abs_min + 1
                return min(top - abs_ofs, size) if abs_ofs < top else 0
        else:
            raise TypeError("Expected 1 or 2 arguments, found %d" % len(args))

    def reshape(self, *args):
        """
        Create a new Decomposition with extended or reduced boundary subdomains.
        This causes a new index enumeration.

        Parameters
        ----------
        *args
            There are three possible cases:
            * int, int. The two integers represent the number of points to remove
              (negative value) or add (positive value) on the left and right sides,
              respectively.
            * slice(a, b). The number of points to remove/add on the left and
              right sides is given by ``self.glb_min - a`` and ``self.glb_max - b``,
              respectively.
            * array_like. Explicitly states what indices in ``self`` should be
              retained.

        Raises
        ------
        TypeError
            If the input doesn't adhere to any of the supported format.

        Examples
        --------
        >>> d = Decomposition([[0, 1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11]], 2)
        >>> d
        Decomposition([0,2], [3,4], <<[5,7]>>, [8,11])

        Providing explicit values

        >>> d.reshape(1, 1)
        Decomposition([0,3], [4,5], <<[6,8]>>, [9,13])

        >>> d.reshape(-2, 2)
        Decomposition([0,0], [1,2], <<[3,5]>>, [6,11])

        Providing a slice

        >>> d.reshape(slice(2, 9))
        Decomposition([0,0], [1,2], <<[3,5]>>, [6,6])

        >>> d.reshape(slice(2, -2))
        Decomposition([0,0], [1,2], <<[3,5]>>, [6,7])

        >>> d.reshape(slice(4))
        Decomposition([0,2], [3,3], <<[]>>, [])
        """

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, slice):
                if arg.start is None or self.glb_min is None:
                    nleft = 0
                else:
                    nleft = self.glb_min - arg.start
                if arg.stop is None or self.glb_max is None:
                    nright = 0
                elif arg.stop < 0:
                    nright = arg.stop
                else:
                    nright = arg.stop - self.glb_max - 1
            elif isinstance(arg, Iterable):
                items = [np.array([j for j in i if j in arg]) for i in self]
                for i, arr in enumerate(list(items)):
                    items[i] = np.arange(arr.size) + sum(j.size for j in items[:i])
                return Decomposition(items, self.local)
        elif len(args) == 2:
            nleft, nright = args
        else:
            raise TypeError("Expected 1 or 2 arguments, found %d" % len(args))

        items = list(self)

        # Handle corner cases first
        if -nleft >= self.size or -nright >= self.size:
            return Decomposition([np.array([])]*len(self), self.local)

        # Handle left extension/reduction
        if nleft > 0:
            items = [np.concatenate([np.arange(-nleft, 0), items[0]])] + items[1:]
        elif nleft < 0:
            n = 0
            for i, sd in enumerate(list(items)):
                if n + sd.size >= -nleft:
                    items = [np.array([])]*i + [sd[(-nleft - n):]] + items[i+1:]
                    break
                n += sd.size

        # Handle right extension/reduction
        if nright > 0:
            extension = np.arange(self.glb_max + 1, self.glb_max + 1 + nright)
            items = items[:-1] + [np.concatenate([items[-1], extension])]
        elif nright < 0:
            n = 0
            for i, sd in enumerate(reversed(list(items))):
                if n + sd.size >= -nright:
                    items = items[:-i-1] + [sd[:(nright + n)]] + [np.array([])]*i
                    break
                n += sd.size

        # Renumbering
        items = [i + nleft for i in items]

        return Decomposition(items, self.local)
