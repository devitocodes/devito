import cgen as c

from devito.dle import compose_nodes, is_foldable, retrieve_iteration_tree
from devito.nodes import Iteration, List
from devito.visitors import (FindAdjacentIterations, IsPerfectIteration,
                             NestedTransformer, Transformer)
from devito.tools import as_tuple

__all__ = ['fold_blockable_tree', 'unfold_blocked_tree']


def fold_blockable_tree(node):
    """
    Create :class:`IterationFold`s from sequences of nested :class:`Iteration`.
    """
    found = FindAdjacentIterations().visit(node)
    found.pop('seen_iteration')

    mapper = {}
    for k, v in found.items():
        for i in v:
            # Check if the Iterations in /i/ are foldable or not
            assert len(i) > 1
            if any(not IsPerfectIteration().visit(j) for j in i):
                continue
            trees = [retrieve_iteration_tree(j)[0] for j in i]
            if any(len(trees[0]) != len(j) for j in trees):
                continue
            pairwise_folds = zip(*reversed(trees))
            if any(not is_foldable(j) for j in pairwise_folds):
                continue
            for j in pairwise_folds:
                root, remainder = j[0], j[1:]
                folds = [(tuple(y-x for x, y in zip(i.offsets, root.offsets)), i.nodes)
                         for i in remainder]
                mapper[root] = IterationFold(folds=folds, **root.args)
                for k in remainder:
                    mapper[k] = None

    # Insert the IterationFolds in the Iteration/Expression tree
    processed = NestedTransformer(mapper).visit(node)

    return processed


def unfold_blocked_tree(node):
    """
    Unfold nested :class:`IterationFold`.

    Examples
    ========
    Given a section of Iteration/Expression tree as below: ::

        for i = 1 to N-1  // folded
          for j = 1 to N-1  // folded
            foo1()

    Assuming a fold with offset 1 in both /i/ and /j/ and body ``foo2()``, create:

        for i = 1 to N-1
          for j = 1 to N-1
            foo1()
        for i = 2 to N-2
          for j = 2 to N-2
            foo2()
    """
    # Search the unfolding candidates
    candidates = []
    for tree in retrieve_iteration_tree(node):
        handle = tuple(i for i in tree if i.is_IterationFold)
        if handle:
            # Sanity check
            assert IsPerfectIteration().visit(handle[0])
            candidates.append(handle)

    # Perform unfolding
    mapper = {}
    for tree in candidates:
        unfolded = zip(*[i.unfold() for i in tree])
        unfolded = [compose_nodes(i) for i in unfolded]
        mapper[tree[0]] = List(body=unfolded)

    # Insert the unfolded Iterations in the Iteration/Expression tree
    processed = Transformer(mapper).visit(node)

    return processed


class IterationFold(Iteration):

    """
    An IterationFold is a special :class:`Iteration` object that represents
    a sequence of consecutive (in program order) Iterations. In an IterationFold,
    all Iterations of the sequence but the so called ``root`` are "hidden"; that is,
    they cannot be visited by an Iteration/Expression tree visitor.

    The Iterations in the sequence represented by the IterationFold all have same
    dimension and properties. However, their extent is relative to that of the ``root``.
    """

    is_IterationFold = True

    def __init__(self, nodes, dimension, limits, index=None, offsets=None,
                 properties=None, folds=None):
        super(IterationFold, self).__init__(nodes, dimension, limits, index,
                                            offsets, properties)
        self.folds = folds

    def __repr__(self):
        properties = ""
        if self.properties:
            properties = "WithProperties[%s]::" % ",".join(self.properties)
        length = "Length %d" % len(self.folds)
        return "<%sIterationFold %s; %s; %s>" % (properties, self.index,
                                                 self.limits, length)

    @property
    def ccode(self):
        comment = c.Comment('This IterationFold is "hiding" ore or more Iterations')
        code = super(IterationFold, self).ccode
        return c.Module([comment, code])

    def unfold(self):
        """
        Return the corresponding :class:`Iteration` objects from each fold in ``self``.
        """
        args = self.args
        args.pop('folds')

        # Construct the root Iteration
        root = Iteration(**args)

        # Construct the folds
        args.pop('nodes')
        args.pop('offsets')
        start, end, incr = args.pop('limits')
        folds = tuple(Iteration(nodes, limits=[start+ofs[0], end+ofs[1], incr], **args)
                      for ofs, nodes in self.folds)

        return folds + as_tuple(root)
