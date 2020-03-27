from collections import OrderedDict
from collections.abc import Iterable, Mapping
from functools import reduce
from operator import attrgetter

from devito.tools.utils import flatten

__all__ = ['toposort']


def build_dependence_lists(elements):
    """
    Given an iterable of dependences, return the dependence lists as a
    mapper suitable for graph-like algorithms. A dependence is an iterable of
    elements ``[a, b, c, ...]``, meaning that ``a`` preceeds ``b`` and ``c``,
    ``b`` preceeds ``c``, and so on.
    """
    mapper = OrderedDict()
    for element in elements:
        for idx, i0 in enumerate(element):
            v = mapper.setdefault(i0, set())
            for i1 in element[idx + 1:]:
                v.add(i1)
    return mapper


def toposort(data):
    """
    Given items that depend on other items, a topological sort arranges items in
    order that no one item precedes an item it depends on.

    ``data`` captures the various dependencies. It may be:

        * A dictionary whose keys are items and whose values are a set of
          dependent items. The dictionary may contain self-dependencies
          (which are ignored), and dependent items that are not also
          dict keys.
        * An iterable of dependences as expected by :func:`build_dependence_lists`.

    Readapted from: ::

        http://code.activestate.com/recipes/577413/
    """
    if not isinstance(data, Mapping):
        assert isinstance(data, Iterable)
        data = build_dependence_lists(data)

    processed = []

    if not data:
        return processed

    # Do not transform `data` in place
    mapper = OrderedDict([(k, set(v)) for k, v in data.items()])

    # Ignore self dependencies
    for k, v in mapper.items():
        v.discard(k)

    # Perform the topological sorting
    extra_items_in_deps = reduce(set.union, mapper.values()) - set(mapper)
    mapper.update(OrderedDict([(item, set()) for item in extra_items_in_deps]))
    while True:
        ordered = set(item for item, dep in mapper.items() if not dep)
        if not ordered:
            break
        try:
            processed = sorted(ordered, key=attrgetter('name')) + processed
        except AttributeError:
            processed = sorted(ordered) + processed
        mapper = OrderedDict([(item, (dep - ordered)) for item, dep in mapper.items()
                              if item not in ordered])
    if len(processed) != len(set(flatten(data) + flatten(data.values()))):
        raise ValueError("A cyclic dependency exists amongst %r" % data)
    return processed
