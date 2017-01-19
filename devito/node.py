"""Base classes for the Iteration/Expression hierarchy."""

from collections import defaultdict

import cgen as c

from devito.tools import as_tuple

__all__ = ['Node', 'Block']


class Node(object):

    is_Node = True
    is_Block = False
    is_Iteration = False
    is_Expression = False

    def __new__(cls, *args, **kwargs):
        obj = super(Node, cls).__new__(cls)
        obj._args = args + (kwargs,)
        return obj

    def _children(self):
        """Return the traversable children."""
        return []

    def _rebuild(self):
        """Reconstruct the Node."""
        raise NotImplementedError()

    def indexify(self):
        """Convert all enclosed nodes to "indexed" format"""
        for e in self._children():
            e.indexify()

    def substitute(self, substitutions):
        """Apply substitutions to children nodes

        :param substitutions: Dict containing the substitutions to apply.
        """
        candidates = [n for n in self._children() if isinstance(n, Node)]
        for n in candidates:
            n.substitute(substitutions)

    @property
    def index_offsets(self):
        """Collect all non-zero offsets used with each index in a map

        Note: This assumes we have indexified the stencil expression."""
        return defaultdict(list)

    @property
    def ccode(self):
        """Generate C code."""
        raise NotImplementedError()

    @property
    def signature(self):
        """List of data objects used by the Node."""
        raise NotImplementedError()

    @property
    def args(self):
        return self._args


class Block(Node):

    is_Block = True

    def __init__(self, header=None, body=None, footer=None):
        self.header = as_tuple(header)
        self.body = as_tuple(body)
        self.footer = as_tuple(footer)

    def __repr__(self):
        header = "".join([str(s) for s in self.header])
        body = "Block::\n\t%s" % "\n\t".join([str(s) for s in self.body])
        footer = "".join([str(s) for s in self.footer])
        return header + body + footer

    @property
    def ccode(self):
        body = tuple(s.ccode for s in self.body)
        return c.Block(self.header + body + self.footer)

    def _children(self):
        return self.header + self.body + self.footer
