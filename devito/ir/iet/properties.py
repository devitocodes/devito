from devito.tools import Tag


class IterationProperty(Tag):

    """
    An Iteration decorator.
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

TILABLE = IterationProperty('tilable')
"""The Iteration can be tiled (or "loop blocking")."""

WRAPPABLE = IterationProperty('wrappable')
"""
The Iteration implements modulo buffered iteration and its expressions are so that
one or more buffer slots can be dropped without affecting correctness. For example,
u[t+1, ...] = f(u[t, ...], u[t-1, ...]) --> u[t-1, ...] = f(u[t, ...], u[t-1, ...]).
"""

ROUNDABLE = IterationProperty('roundable')
"""
The Iteration writes (only) to Arrays and the trip count can be rounded up to a
multiple of the SIMD vector length without affecting correctness (thanks to the
presence of sufficient padding).
"""

AFFINE = IterationProperty('affine')
"""
All Indexed access functions using the Iteration dimension ``d`` are
affine in ``d``. Further, the Iteration does not contain any Indexed varying in
``d`` used to indirectly access some other Indexed.
"""


class HaloSpotProperty(Tag):

    """
    A HaloSpot decorator.
    """

    pass


OVERLAPPABLE = HaloSpotProperty('overlappable')
"""The HaloSpot supports computation-communication overlap."""


def hoistable(i):
    """
    The HaloSpot can be squashed with a previous HaloSpot as all data dependences
    would still be honored.
    """
    return HaloSpotProperty('hoistable', i)


def useless(i):
    """
    The HaloSpot can be ignored as a halo update at this point would be completely
    useless.
    """
    return HaloSpotProperty('useless', i)
