import numbers
from cached_property import cached_property

from devito.ir.iet import nodes
from devito.types.basic import AbstractFunction

from devito.ops.types import String


class Call(nodes.Call):

    @cached_property
    def free_symbols(self):
        free = set()
        for i in self.arguments:
            if isinstance(i, (numbers.Number, String)):
                continue
            elif isinstance(i, AbstractFunction):
                free.add(i)
            else:
                free.update(i.free_symbols)
        return free
