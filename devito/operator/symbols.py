from devito.tools import generator

__init__ = ['SymbolRegistry']


class SymbolRegistry(object):

    """A registry for all the symbols used by an Operator."""

    _symbol_prefix = 'r'

    def __init__(self, nthreads=None, nthreads_nested=None, nthreads_nonaffine=None,
                 threadid=None):
        # {name -> generator()} -- to create unique names for symbols, functions, ...
        self.counters = {}

        # Special symbols
        self.nthreads = nthreads
        self.nthreads_nested = nthreads_nested
        self.nthreads_nonaffine = nthreads_nonaffine
        self.threadid = threadid

    def make_name(self, prefix=None):
        # By default we're creating a new symbol
        if prefix is None:
            prefix = self._symbol_prefix

        try:
            counter = self.counters[prefix]
        except KeyError:
            counter = self.counters.setdefault(prefix, generator())

        return "%s%d" % (prefix, counter())
