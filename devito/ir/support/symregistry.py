from devito.tools import generator
from devito.types import (DeviceID, NThreads, NThreadsNested, NThreadsNonaffine,
                          NPThreads, ThreadID)

__init__ = ['SymbolRegistry']


class SymbolRegistry(object):

    """A registry for all the symbols used by an Operator."""

    _symbol_prefix = 'r'

    def __init__(self):
        # {name -> generator()} -- to create unique names for symbols, functions, ...
        self.counters = {}

        # Special symbols
        self.nthreads = NThreads()
        self.nthreads_nested = NThreadsNested()
        self.nthreads_nonaffine = NThreadsNonaffine()
        self.threadid = ThreadID(self.nthreads)
        self.deviceid = DeviceID()

        # Several groups of pthreads each of size `npthread` may be created
        # during compilation
        self.npthreads = []

        # A local cache is used to track symbols across different compiler
        # passes, to maximize symbol (especially Dimension) reuse
        self.caches = {}

    def make_name(self, prefix=None):
        # By default we're creating a new symbol
        if prefix is None:
            prefix = self._symbol_prefix

        try:
            counter = self.counters[prefix]
        except KeyError:
            counter = self.counters.setdefault(prefix, generator())

        return "%s%d" % (prefix, counter())

    def make_npthreads(self, size):
        name = self.make_name(prefix='npthreads')
        npthreads = NPThreads(name=name, size=size)
        self.npthreads.append(npthreads)
        return npthreads

    def get(self, cachename, key):
        """
        Get a symbol from one of the local caches.
        """
        return self.caches[cachename][key]

    def setdefault(self, cachename, key, v):
        """
        Like dict.setdefault, but applied to one of the local caches.
        """
        return self.caches.setdefault(cachename, {}).setdefault(key, v)
