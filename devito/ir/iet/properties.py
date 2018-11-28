from devito.tools import Tag


class IterationProperty(Tag):

    """
    An :class:`Iteration` decorator.
    """

    _KNOWN = []

    def __init__(self, name, val=None):
        super(IterationProperty, self).__init__(name, val)
        IterationProperty._KNOWN.append(self)


SEQUENTIAL = IterationProperty('sequential')
"""The Iteration is inherently serial, i.e., its iterations cannot run in parallel."""

PARALLEL = IterationProperty('parallel')
"""The Iteration can be executed in parallel w/o need for synchronization."""

PARALLEL_IF_ATOMIC = IterationProperty('parallel_if_atomic')
"""The Iteration can be executed in parallel as long as all increments are
guaranteed to be atomic."""

COLLAPSED = lambda i: IterationProperty('collapsed', i)
"""The Iteration is the root of a nest of ``i`` collapsed Iterations."""

VECTOR = IterationProperty('vector-dim')
"""The Iteration can be SIMD-vectorized."""

ELEMENTAL = IterationProperty('elemental')
"""The Iteration can be pulled out to an elemental function."""

REMAINDER = IterationProperty('remainder')
"""The Iteration implements a remainder/peeler loop."""

WRAPPABLE = IterationProperty('wrappable')
"""The Iteration implements modulo buffered iteration and its expressions are so that
one or more buffer slots can be dropped without affecting correctness. For example,
u[t+1, ...] = f(u[t, ...], u[t-1, ...]) --> u[t-1, ...] = f(u[t, ...], u[t-1, ...])."""

AFFINE = IterationProperty('affine')
"""All :class:`Indexed`s' access functions using the Iteration dimension ``d`` are
affine in ``d``. Further, the Iteration does not contain any Indexed varying in
``d`` used to indirectly access some other Indexed."""


def tagger(i):
    return IterationProperty('tag', i)


def ntags():
    return len(IterationProperty._KNOWN) - ntags.n_original_properties
ntags.n_original_properties = len(IterationProperty._KNOWN)  # noqa


class HaloSpotProperty(Tag):

    """
    A :class:`HaloSpot` decorator.
    """

    pass


REDUNDANT = HaloSpotProperty('redundant')
"""The HaloSpot is redundant given that some other HaloSpots already take care
of updating the data accessed in the sub-tree."""
