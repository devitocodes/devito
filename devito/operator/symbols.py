from devito.tools import generator
from devito.types import NThreads, NThreadsNested, NThreadsNonaffine, NPThreads, ThreadID

__init__ = ['SymbolRegistry']


class SymbolRegistry(object):

    """A registry for all the symbols used by an Operator."""

    _symbol_prefix = 'r'

    def __init__(self):
        # {name -> generator()} -- to create unique names for symbols, functions, ...
        self.counters = {}

        # Special symbols
        self.nthreads = NThreads(aliases='nthreads0')
        self.nthreads_nested = NThreadsNested(aliases='nthreads1')
        self.nthreads_nonaffine = NThreadsNonaffine(aliases='nthreads2')
        self.threadid = ThreadID(self.nthreads)

        # Several groups of pthreads each of size `npthread` may be created
        # during compilation
        self.npthreads = []

    def make_name(self, prefix=None):
        # By default we're creating a new symbol
        if prefix is None:
            prefix = self._symbol_prefix

        try:
            counter = self.counters[prefix]
        except KeyError:
            counter = self.counters.setdefault(prefix, generator())

        return "%s%d" % (prefix, counter())

    def make_npthreads(self, value):
        name = self.make_name(prefix='npthreads')
        npthreads = NPThreads(name=name, value=value)
        self.npthreads.append(npthreads)
        return npthreads
