from collections.abc import Iterable

from itertools import groupby

from devito.ir.support import IterationSpace, null_ispace
from devito.tools import flatten, timed_pass

__all__ = ['Queue', 'cluster_pass']


class Queue:

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
    _q_syncs_in_key = False

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
            properties = cluster.properties.drop(cluster.ispace[level:].itdims)
        else:
            properties = None

        if self._q_syncs_in_key:
            try:
                syncs = tuple(cluster.syncs.get(i.dim) for i in ispace)
            except AttributeError:
                # `cluster` is actually a ClusterGroup
                assert len(cluster.syncs) == 1
                syncs = tuple(cluster.syncs[0].get(i.dim) for i in ispace)
        else:
            syncs = None

        prefix = Prefix(ispace, guards, properties, syncs)

        subkey = self._make_key_hook(cluster, level)

        return (prefix,) + subkey

    def _make_key_hook(self, cluster, level):
        return ()

    def _process_fdta(self, clusters, level, prefix=null_ispace, **kwargs):
        """
        fdta -> First Divide Then Apply
        """
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


class Prefix(IterationSpace):

    def __init__(self, ispace, guards, properties, syncs):
        super().__init__(ispace.intervals, ispace.sub_iterators, ispace.directions)

        self.guards = guards
        self.properties = properties
        self.syncs = syncs

    def __eq__(self, other):
        return (isinstance(other, Prefix) and
                super().__eq__(other) and
                self.guards == other.guards and
                self.properties == other.properties and
                self.syncs == other.syncs)

    def __hash__(self):
        return hash((self.intervals, self.sub_iterators, self.directions,
                     self.guards, self.properties, self.syncs))


class cluster_pass:

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
            self.cond = lambda c: (c.is_dense or not c.is_sparse) and not c.is_wild
        elif mode == 'sparse':
            self.cond = lambda c: c.is_sparse and not c.is_wild
        else:
            self.cond = lambda c: True

    def __call__(self, *args, **kwargs):
        if timed_pass.is_enabled():
            maybe_timed = lambda *_args: \
                timed_pass(self.func, self.func.__name__)(*_args, **kwargs)
        else:
            maybe_timed = lambda *_args: self.func(*_args, **kwargs)
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
