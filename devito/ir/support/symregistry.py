from devito.symbolics import retrieve_functions
from devito.tools import as_tuple, generator
from devito.types import (
    DeviceID, NPThreads, NThreads, NThreadsNested, NThreadsNonaffine, ThreadID
)

__init__ = ['SymbolRegistry']


class SymbolRegistry:

    """A registry for all the symbols used by an Operator."""

    _symbol_prefix = 'r'

    def __init__(self):
        # {name -> generator()} -- to create unique names for symbols, functions, ...
        self.counters = {}
        self.reserved = set()

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

    def reserve(self, exprs):
        """Reserve identifiers already used by the input expressions."""
        exprs = as_tuple(exprs)
        self.reserved.update(i.name for e in exprs for i in e.free_symbols)

        # Before indexification, a Function's free symbols omit its own name
        self.reserved.update(i.name for i in retrieve_functions(exprs))

    def make_name(self, prefix=None, increment_first=True):
        # By default we're creating a new symbol
        if prefix is None:
            prefix = self._symbol_prefix

        try:
            counter = self.counters[prefix]
        except KeyError:
            counter = self.counters.setdefault(prefix, generator())

            # Only increment symbol names after the first encountered
            if not increment_first and prefix not in self.reserved:
                self.reserved.add(prefix)
                return prefix

        name = f'{prefix}{counter()}'
        while name in self.reserved:
            name = f'{prefix}{counter()}'
        self.reserved.add(name)
        return name

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
