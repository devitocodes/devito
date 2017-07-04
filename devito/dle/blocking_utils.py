import cgen as c
from sympy import Eq, Symbol
import numpy as np

from devito.dle import compose_nodes, is_foldable, retrieve_iteration_tree
from devito.dse import xreplace_indices
from devito.nodes import Expression, Iteration, List, LocalExpression
from devito.visitors import (FindAdjacentIterations, FindNodes, IsPerfectIteration,
                             NestedTransformer, Transformer)
from devito.tools import as_tuple

__all__ = ['fold_blockable_tree', 'unfold_blocked_tree']


def fold_blockable_tree(node, exclude_innermost=False):
    """
    Create :class:`IterationFold`s from sequences of nested :class:`Iteration`.
    """
    found = FindAdjacentIterations().visit(node)
    found.pop('seen_iteration')

    mapper = {}
    for k, v in found.items():
        for i in v:
            # Pre-condition: they all must be perfect iterations
            assert len(i) > 1
            if any(not IsPerfectIteration().visit(j) for j in i):
                continue
            # Only retain consecutive trees having same depth
            trees = [retrieve_iteration_tree(j)[0] for j in i]
            handle = []
            for j in trees:
                if len(j) != len(trees[0]):
                    break
                handle.append(j)
            trees = handle
            if not trees:
                continue
            # Check foldability
            pairwise_folds = zip(*reversed(trees))
            if any(not is_foldable(j) for j in pairwise_folds):
                continue
            # Perform folding
            for j in pairwise_folds[:-exclude_innermost]:
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
        trees = zip(*[i.unfold() for i in tree])
        trees = optimize_unfolded_tree(trees[:-1], trees[-1])
        trees = [compose_nodes(i) for i in trees]
        mapper[tree[0]] = List(body=trees)

    # Insert the unfolded Iterations in the Iteration/Expression tree
    processed = Transformer(mapper).visit(node)

    return processed


def optimize_unfolded_tree(unfolded, root):
    """
    Transform folded trees to reduce the memory footprint.

    Examples
    ========
    Given:

        .. code-block::
            for i = 1 to N - 1  # Folded tree
              for j = 1 to N - 1
                tmp[i,j] = ...
            for i = 2 to N - 2  # Root
              for j = 2 to N - 2
                ... = ... tmp[i,j] ...

    The temporary ``tmp`` has shape ``(N-1, N-1)``. However, as soon as the
    iteration space is blocked, with blocks of shape ``(i_bs, j_bs)``, the
    ``tmp`` shape can be shrunk to ``(i_bs-1, j_bs-1)``. The resulting
    iteration tree becomes:

        .. code-block::
            for i = 1 to i_bs + 1  # Folded tree
              for j = 1 to j_bs + 1
                i' = i + i_block - 2
                j' = j + j_block - 2
                tmp[i,j] = ... # use i' and j'
            for i = i_block to i_block + i_bs  # Root
              for j = j_block to j_block + j_bs
                i' = i - x_block
                j' = j - j_block
                ... = ... tmp[i',j'] ...
    """
    processed = []
    for i, tree in enumerate(unfolded):
        otree = []
        stmts = []
        mapper = {}

        # "Shrink" the iteration space
        for j in tree:
            start, end, incr = j.args['limits']
            otree.append(j._rebuild(limits=[0, end-start, incr]))
            index = Symbol('%ss%d' % (j.index, i))
            stmts.append((LocalExpression(Eq(index, j.dim + start), np.int32),
                          LocalExpression(Eq(index, j.dim - start), np.int32)))
            mapper[j.dim] = index

        # Substitute iteration variables within the folded trees
        exprs = FindNodes(Expression).visit(otree[-1])
        replaced = xreplace_indices([j.expr for j in exprs], mapper, only_rhs=True)
        subs = [j._rebuild(expr=k) for j, k in zip(exprs, replaced)]

        handle = Transformer(dict(zip(exprs, subs))).visit(otree[-1])
        handle = handle._rebuild(nodes=(zip(*stmts)[0] + handle.nodes))
        processed.append(tuple(otree[:-1]) + (handle,))

        # Temporary arrays can now be moved to the stack
        if all(not j.is_Remainder for j in otree):
            shape = tuple(j.bounds_symbolic[1] for j in otree)
            for j in subs:
                shape += j.output_function.shape[len(otree):]
                j.output_function.update(shape=shape, onstack=True)

        # Introduce the new iteration variables within root
        candidates = [j.output for j in subs]
        exprs = FindNodes(Expression).visit(root[-1])
        replaced = xreplace_indices([j.expr for j in exprs], mapper, candidates)
        subs = [j._rebuild(expr=k) for j, k in zip(exprs, replaced)]
        handle = Transformer(dict(zip(exprs, subs))).visit(root[-1])
        handle = handle._rebuild(nodes=(zip(*stmts)[1] + handle.nodes))
        root = root[:-1] + (handle,)

    return processed + [root]


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
                 properties=None, pragmas=None, folds=None):
        super(IterationFold, self).__init__(nodes, dimension, limits, index,
                                            offsets, properties, pragmas)
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
        comment = c.Comment('This IterationFold is "hiding" one or more Iterations')
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
