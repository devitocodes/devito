from anytree import NodeMixin, PostOrderIter, RenderTree, ContStyle

__all__ = ["ScheduleTree", "NodeSection", "NodeIteration", "NodeConditional",
           "NodeExprs", "NodeHalo"]


class ScheduleTree(NodeMixin):

    is_Section = False
    is_Iteration = False
    is_Conditional = False
    is_Exprs = False
    is_Halo = False

    def __init__(self, parent=None):
        self.parent = parent

    def __repr__(self):
        return render(self)

    def visit(self):
        for i in PostOrderIter(self):
            yield i

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

    def __init__(self, ispace, parent=None):
        super(NodeIteration, self).__init__(parent)
        self.ispace = ispace

    @property
    def interval(self):
        return self.ispace.intervals[0]

    @property
    def dim(self):
        return self.interval.dim

    @property
    def limits(self):
        return self.interval.limits

    @property
    def direction(self):
        return self.ispace.directions[self.dim]

    @property
    def sub_iterators(self):
        return self.ispace.sub_iterators.get(self.dim, [])

    @property
    def __repr_render__(self):
        return "%s%s" % (self.dim, self.direction)


class NodeConditional(ScheduleTree):

    is_Conditional = True

    def __init__(self, guard, parent=None):
        super(NodeConditional, self).__init__(parent)
        self.guard = guard

    @property
    def __repr_render__(self):
        return "If"


class NodeExprs(ScheduleTree):

    is_Exprs = True

    def __init__(self, exprs, ispace, dspace, shape, ops, traffic, parent=None):
        super(NodeExprs, self).__init__(parent)
        self.exprs = exprs
        self.ispace = ispace
        self.dspace = dspace
        self.shape = shape
        self.ops = ops
        self.traffic = traffic

    @property
    def __repr_render__(self):
        ths = 2
        n = len(self.exprs)
        ret = ",".join("Eq" for i in range(min(n, ths)))
        ret = ("%s,..." % ret) if n > ths else ret
        return "[%s]" % ret


class NodeHalo(ScheduleTree):

    is_Halo = True

    def __init__(self, halo_scheme):
        self.halo_scheme = halo_scheme

    @property
    def __repr_render__(self):
        return "<Halo>"


def insert(node, parent, children):
    """
    Insert ``node`` between ``parent`` and ``children``, where ``children``
    are a subset of nodes in ``parent.children``.
    """
    processed = []
    for n in list(parent.children):
        if n in children:
            n.parent = node
            if node not in processed:
                processed.append(node)
        else:
            processed.append(n)
    parent.children = processed


def render(stree):
    return RenderTree(stree, style=ContStyle()).by_attr('__repr_render__')
