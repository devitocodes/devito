class IterationProperty(object):

    _KNOWN = []

    """
    A :class:`Iteration` decorator.
    """

    def __init__(self, name, val=None):
        self.name = name
        self.val = val

        self._KNOWN.append(self)

    def __eq__(self, other):
        if not isinstance(other, IterationProperty):
            return False
        return self.name == other.name and self.val == other.val

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, self.val))

    def __str__(self):
        return self.name if self.val is None else '%s%s' % (self.name, str(self.val))

    def __repr__(self):
        if self.val is None:
            return "Property: %s" % self.name
        else:
            return "Property: %s[%s]" % (self.name, str(self.val))


SEQUENTIAL = IterationProperty('sequential')
"""The Iteration is inherently serial, i.e., its iterations cannot run in parallel."""

PARALLEL = IterationProperty('parallel')
"""The Iteration can be executed in parallel w/o need for synchronization."""

PARALLEL_IF_ATOMIC = IterationProperty('parallel_if_atomic')
"""The Iteration can be executed in parallel as long as all increments are
guaranteed to be atomic."""

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


def tagger(i):
    return IterationProperty('tag', i)


def ntags():
    return len(IterationProperty._KNOWN) - ntags.n_original_properties
ntags.n_original_properties = len(IterationProperty._KNOWN)  # noqa
