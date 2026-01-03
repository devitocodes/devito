from anytree import ContStyle, NodeMixin, PostOrderIter, RenderTree

from devito.ir.support import PrefetchUpdate, WithLock

__all__ = [
    "NodeConditional",
    "NodeExprs",
    "NodeHalo",
    "NodeIteration",
    "NodeSection",
    "NodeSync",
    "ScheduleTree",
]


class ScheduleTree(NodeMixin):

    is_Section = False
    is_Iteration = False
    is_Conditional = False
    is_Sync = False
    is_Exprs = False
    is_Halo = False

    def __init__(self, parent=None):
        self.parent = parent

    def __repr__(self):
        return render(self)

    def visit(self):
        yield from PostOrderIter(self)

    @property
    def last(self):
        return self.children[-1] if self.children else None


class NodeSection(ScheduleTree):

    is_Section = True

    @property
    def __repr_render__(self):
        return "<S>"


class NodeIteration(ScheduleTree):

    is_Iteration = True

    def __init__(self, ispace, parent=None, properties=None):
        super().__init__(parent)
        self.ispace = ispace
        self.properties = properties

    @property
    def interval(self):
        return self.ispace.intervals[0]

    @property
    def dim(self):
        return self.interval.dim

    @property
    def limits(self):
        return (self.dim.symbolic_min + self.interval.lower,
                self.dim.symbolic_max + self.interval.upper,
                self.dim.symbolic_incr)

    @property
    def direction(self):
        return self.ispace.directions[self.dim]

    @property
    def sub_iterators(self):
        return self.ispace.sub_iterators.get(self.dim, [])

    @property
    def __repr_render__(self):
        return f"{self.dim}{self.direction}"


class NodeConditional(ScheduleTree):

    is_Conditional = True

    def __init__(self, guard, parent=None):
        super().__init__(parent)
        self.guard = guard

    @property
    def __repr_render__(self):
        return "If"


class NodeSync(ScheduleTree):

    is_Sync = True

    def __init__(self, sync_ops, parent=None):
        super().__init__(parent)
        self.sync_ops = sync_ops

    @property
    def __repr_render__(self):
        return "Sync[{}]".format(",".join(i.__class__.__name__ for i in self.sync_ops))

    @property
    def is_async(self):
        return any(isinstance(i, (WithLock, PrefetchUpdate)) for i in self.sync_ops)


class NodeExprs(ScheduleTree):

    is_Exprs = True

    def __init__(self, exprs, ispace, dspace, ops, traffic, parent=None):
        super().__init__(parent)
        self.exprs = exprs
        self.ispace = ispace
        self.dspace = dspace
        self.ops = ops
        self.traffic = traffic

    @property
    def __repr_render__(self):
        threshold = 2
        n = len(self.exprs)
        ret = ",".join("Eq" for i in range(min(n, threshold)))
        ret = (f"{ret},...") if n > threshold else ret
        return f"[{ret}]"


class NodeHalo(ScheduleTree):

    is_Halo = True

    def __init__(self, halo_scheme, parent=None):
        super().__init__(parent)
        self.halo_scheme = halo_scheme

    @property
    def __repr_render__(self):
        return "<Halo>"


def render(stree):
    return RenderTree(stree, style=ContStyle()).by_attr('__repr_render__')
