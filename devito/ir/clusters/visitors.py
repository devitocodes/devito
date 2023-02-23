from collections import defaultdict
from collections.abc import Iterable

from itertools import groupby

from devito.ir.support import IterationSpace, Scope
from devito.tools import as_tuple, flatten, timed_pass

__all__ = ['Queue', 'QueueStateful', 'cluster_pass']


class Queue(object):

    """
    A special queue to process Clusters based on a divide-and-conquer algorithm.

    Notes
    -----
    Subclasses must override :meth:`callback`, which may get executed either
    before (fdta -- first divide then apply) or after (fatd -- first apply
    then divide) the divide phase of the algorithm.
    """

    # Handlers for the construction of the key used in the visit
    # Some visitors may need a relaxed key to process together certain groups
    # of Clusters
    _q_ispace_in_key = True
    _q_guards_in_key = False
    _q_properties_in_key = False

    def callback(self, *args):
        raise NotImplementedError

    def process(self, clusters):
        return self._process_fdta(clusters, 1)

    def _make_key(self, cluster, level):
        assert self._q_ispace_in_key
        ispace = cluster.ispace[:level]

        if self._q_guards_in_key:
            try:
                guards = tuple(cluster.guards.get(i.dim) for i in ispace)
            except AttributeError:
                # `cluster` is actually a ClusterGroup
                assert len(cluster.guards) == 1
                guards = tuple(cluster.guards[0].get(i.dim) for i in ispace)
        else:
            guards = None

        if self._q_properties_in_key:
            properties = cluster.properties.drop(cluster.ispace[level:].itdimensions)
        else:
            properties = None

        prefix = Prefix(ispace, guards, properties)

        subkey = self._make_key_hook(cluster, level)

        return (prefix,) + subkey

    def _make_key_hook(self, cluster, level):
        return ()

    def _process_fdta(self, clusters, level, prefix=None, **kwargs):
        """
        fdta -> First Divide Then Apply
        """
        prefix = prefix or IterationSpace([])

        # Divide part
        processed = []
        for k, g in groupby(clusters, key=lambda i: self._make_key(i, level)):
            pfx = k[0]
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Recursion
                processed.extend(self._process_fdta(list(g), level + 1, pfx, **kwargs))

        # Apply callback
        processed = self.callback(processed, prefix, **kwargs)

        return processed

    def _process_fatd(self, clusters, level, prefix=None, **kwargs):
        """
        fatd -> First Apply Then Divide
        """
        # Divide part
        processed = []
        for k, g in groupby(clusters, key=lambda i: self._make_key(i, level)):
            pfx = k[0]
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Apply callback
                _clusters = self.callback(list(g), pfx, **kwargs)
                # Recursion
                processed.extend(self._process_fatd(_clusters, level + 1, pfx, **kwargs))

        return processed


class QueueStateful(Queue):

    """
    A Queue carrying along some state. This is useful when one wants to avoid
    expensive re-computations of information.
    """

    class State(object):

        def __init__(self):
            self.properties = {}
            self.scopes = {}

    def __init__(self, state=None):
        super(QueueStateful, self).__init__()
        self.state = state or QueueStateful.State()

    def _fetch_scope(self, clusters):
        exprs = flatten(c.exprs for c in as_tuple(clusters))
        key = tuple(exprs)
        if key not in self.state.scopes:
            self.state.scopes[key] = Scope(exprs)
        return self.state.scopes[key]

    def _fetch_properties(self, clusters, prefix):
        # If the situation is:
        #
        # t
        #   x0
        #     <some clusters>
        #   x1
        #     <some other clusters>
        #
        # then retain only the "common" properties, that is those along `t`
        properties = defaultdict(set)
        for c in clusters:
            v = self.state.properties.get(c, {})
            for i in prefix:
                properties[i.dim].update(v.get(i.dim, set()))
        return properties


class Prefix(IterationSpace):

    def __init__(self, ispace, guards, properties):
        super().__init__(ispace.intervals, ispace.sub_iterators, ispace.directions)

        self.guards = guards
        self.properties = properties

    def __eq__(self, other):
        return (isinstance(other, Prefix) and
                super().__eq__(other) and
                self.guards == other.guards and
                self.properties == other.properties)

    def __hash__(self):
        return hash((self.intervals, self.sub_iterators, self.directions,
                     self.guards, self.properties))


class cluster_pass(object):

    def __new__(cls, *args, mode='dense'):
        if args:
            if len(args) == 1:
                func, = args
            elif len(args) == 2:
                func, mode = args
            else:
                assert False
            obj = object.__new__(cls)
            obj.__init__(func, mode)
            return obj
        else:
            def wrapper(func):
                return cluster_pass(func, mode)
            return wrapper

    def __init__(self, func, mode='dense'):
        self.func = func

        if mode == 'dense':
            self.cond = lambda c: c.is_dense
        elif mode == 'sparse':
            self.cond = lambda c: not c.is_dense
        else:
            self.cond = lambda c: True

    def __call__(self, *args):
        if timed_pass.is_enabled():
            maybe_timed = lambda *_args: timed_pass(self.func, self.func.__name__)(*_args)
        else:
            maybe_timed = lambda *_args: self.func(*_args)
        args = list(args)
        maybe_clusters = args.pop(0)
        if isinstance(maybe_clusters, Iterable):
            # Instance method
            processed = [maybe_timed(c, *args) if self.cond(c) else c
                         for c in maybe_clusters]
        else:
            # Pure function
            self = maybe_clusters
            clusters = args.pop(0)
            processed = [maybe_timed(self, c, *args) if self.cond(c) else c
                         for c in clusters]
        return flatten(processed)
