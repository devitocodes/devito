class UnboundedIndex(object):

    """
    A generic loop iteration index that can be used in a :class:`Iteration` to
    add a non-linear traversal of the iteration space.
    """

    def __init__(self, index, start=0, step=None):
        self.index = index
        self.start = start
        self.step = index + 1 if step is None else step
