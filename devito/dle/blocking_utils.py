from itertools import groupby

import cgen as c
import numpy as np
from cached_property import cached_property

from devito.ir.iet import (Expression, Iteration, List, FindAdjacent,
                           FindNodes, IsPerfectIteration, Transformer,
                           compose_nodes, retrieve_iteration_tree)
from devito.logger import warning
from devito.symbolics import as_symbol, xreplace_indices
from devito.tools import as_tuple, flatten
from devito.types import IncrDimension, Scalar

__all__ = ['BlockDimension', 'fold_blockable_tree', 'unfold_blocked_tree']


def fold_blockable_tree(iet, blockinner=True):
    """
    Create IterationFolds from sequences of nested Iterations.
    """
    mapper = {}
    for k, sequence in FindAdjacent(Iteration).visit(iet).items():
        # Group based on Dimension
        groups = []
        for subsequence in sequence:
            for _, v in groupby(subsequence, lambda i: i.dim):
                i = list(v)
                if len(i) >= 2:
                    groups.append(i)
        for i in groups:
            # Pre-condition: they all must be perfect iterations
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
            pairwise_folds = list(zip(*reversed(trees)))
            if any(not is_foldable(j) for j in pairwise_folds):
                continue
            # Maybe heuristically exclude innermost Iteration
            if blockinner is False:
                pairwise_folds = pairwise_folds[:-1]
            # Perhaps there's nothing to fold
            if len(pairwise_folds) == 0:
                continue
            # TODO: we do not currently support blocking if any of the foldable
            # iterations writes to user data (need min/max loop bounds?)
            exprs = flatten(FindNodes(Expression).visit(j.root) for j in trees[:-1])
            if any(j.write.is_Input for j in exprs):
                continue
            # Perform folding
            for j in pairwise_folds:
                r, remainder = j[0], j[1:]
                folds = [(tuple(y-x for x, y in zip(i.offsets, r.offsets)), i.nodes)
                         for i in remainder]
                mapper[r] = IterationFold(folds=folds, **r.args)
                for k in remainder:
                    mapper[k] = None

    # Insert the IterationFolds in the Iteration/Expression tree
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


def unfold_blocked_tree(iet):
    """
    Unfold nested IterationFolds.

    Examples
    --------

    Given a section of Iteration/Expression tree as below: ::

        for i = 1 to N-1  // folded
          for j = 1 to N-1  // folded
            foo1()

    Assuming a fold with offset 1 in both /i/ and /j/ and body ``foo2()``, create: ::

        for i = 1 to N-1
          for j = 1 to N-1
            foo1()
        for i = 2 to N-2
          for j = 2 to N-2
            foo2()
    """
    # Search the unfolding candidates
    candidates = []
    for tree in retrieve_iteration_tree(iet):
        handle = tuple(i for i in tree if i.is_IterationFold)
        if handle:
            # Sanity check
            assert IsPerfectIteration().visit(handle[0])
            candidates.append(handle)

    # Perform unfolding
    mapper = {}
    for tree in candidates:
        trees = list(zip(*[i.unfold() for i in tree]))
        trees = optimize_unfolded_tree(trees[:-1], trees[-1])
        mapper[tree[0]] = List(body=trees)

    # Insert the unfolded Iterations in the Iteration/Expression tree
    iet = Transformer(mapper).visit(iet)

    return iet


def is_foldable(nodes):
    """
    Return True if the iterable ``nodes`` consists of foldable Iterations,
    False otherwise.
    """
    nodes = as_tuple(nodes)
    if len(nodes) <= 1 or any(not i.is_Iteration for i in nodes):
        return False
    main = nodes[0]
    return all(i.dim == main.dim and i.limits == main.limits and i.index == main.index
               and i.properties == main.properties for i in nodes)


def optimize_unfolded_tree(unfolded, root):
    """
    Transform folded trees to reduce the memory footprint.

    Examples
    --------
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
        assert len(tree) == len(root)

        # We can optimize the folded trees only if they compute temporary
        # arrays, but not if they compute input data
        exprs = FindNodes(Expression).visit(tree[-1])
        writes = [j.write for j in exprs if j.is_tensor]
        if not all(j.is_Array for j in writes):
            processed.append(compose_nodes(tree))
            root = compose_nodes(root)
            continue

        # Shrink the iteration space
        modified_tree = []
        modified_root = []
        modified_dims = {}
        mapper = {}
        for t, r in zip(tree, root):
            udim0 = IncrDimension(t.dim, t.symbolic_min, 1, "%ss%d" % (t.index, i))
            modified_tree.append(t._rebuild(limits=(0, t.limits[1] - t.limits[0], t.step),
                                            uindices=t.uindices + (udim0,)))

            mapper[t.dim] = udim0

            udim1 = IncrDimension(t.dim, 0, 1, "%ss%d" % (t.index, i))
            modified_root.append(r._rebuild(uindices=r.uindices + (udim1,)))

            d = r.limits[0]
            assert isinstance(d, BlockDimension)
            modified_dims[d.root] = d

        # Temporary arrays can now be moved onto the stack
        for w in writes:
            dims = tuple(modified_dims.get(d, d) for d in w.dimensions)
            shape = tuple(d.symbolic_size for d in dims)
            w.update(shape=shape, dimensions=dims, scope='stack')

        # Substitute iteration variables within the folded trees
        modified_tree = compose_nodes(modified_tree)
        replaced = xreplace_indices([j.expr for j in exprs], mapper, only_rhs=True)
        subs = [j._rebuild(expr=k) for j, k in zip(exprs, replaced)]
        processed.append(Transformer(dict(zip(exprs, subs))).visit(modified_tree))

        # Introduce the new iteration variables within /root/
        modified_root = compose_nodes(modified_root)
        exprs = FindNodes(Expression).visit(modified_root)
        candidates = [as_symbol(j.output) for j in subs]
        replaced = xreplace_indices([j.expr for j in exprs], mapper, candidates)
        subs = [j._rebuild(expr=k) for j, k in zip(exprs, replaced)]
        root = Transformer(dict(zip(exprs, subs))).visit(modified_root)

    return processed + [root]


class IterationFold(Iteration):

    """
    An IterationFold is a special Iteration object that represents a sequence of
    consecutive (in program order) Iterations. In an IterationFold, all Iterations
    of the sequence but the so called ``root`` are "hidden"; that is, they cannot
    be visited by an Iteration/Expression tree visitor.

    The Iterations in the sequence represented by the IterationFold all have same
    dimension and properties. However, their extent is relative to that of the
    ``root``.
    """

    is_IterationFold = True

    def __init__(self, *args, **kwargs):
        self.folds = kwargs.pop('folds', None)
        super(IterationFold, self).__init__(*args, **kwargs)

    def __repr__(self):
        properties = ""
        if self.properties:
            properties = [str(i) for i in self.properties]
            properties = "WithProperties[%s]::" % ",".join(properties)
        index = self.index
        if self.uindices:
            index += '[%s]' % ','.join(i.name for i in self.uindices)
        length = "Length %d" % len(self.folds)
        return "<%sIterationFold %s; %s; %s>" % (properties, index, self.limits, length)

    @property
    def ccode(self):
        comment = c.Comment('This IterationFold is "hiding" one or more Iterations')
        code = super(IterationFold, self).ccode
        return c.Module([comment, code])

    def unfold(self):
        """Return an unfolded sequence of Iterations."""
        args = self.args
        args.pop('folds')

        # Construct the root Iteration
        root = Iteration(**args)

        # Construct the folds
        args.pop('nodes')
        ofs = args.pop('offsets')
        try:
            _min, _max, incr = args.pop('limits')
        except TypeError:
            _min, _max, incr = self.limits
        folds = tuple(Iteration(nodes, limits=(_min, _max, incr),
                                offsets=tuple(i-j for i, j in zip(ofs, shift)), **args)
                      for shift, nodes in self.folds)

        return folds + as_tuple(root)


class BlockDimension(IncrDimension):

    @cached_property
    def symbolic_min(self):
        return Scalar(name=self.min_name, dtype=np.int32, is_const=True)

    @property
    def _arg_names(self):
        return (self.step.name,) + self.parent._arg_names

    def _arg_defaults(self, **kwargs):
        # TODO: need a heuristic to pick a default block size
        return {self.step.name: 8}

    def _arg_values(self, args, interval, grid, **kwargs):
        if self.step.name in kwargs:
            value = kwargs.pop(self.step.name)
            if value <= args[self.root.max_name] - args[self.root.min_name] + 1:
                return {self.step.name: value}
            elif value < 0:
                raise ValueError("Illegale block size `%s=%d` (it should be > 0)"
                                 % (self.step.name, value))
            else:
                # Avoid OOB
                warning("The specified block size `%s=%d` is bigger than the "
                        "iteration range; shrinking it to `%s=1`."
                        % (self.step.name, value, self.step.name))
                return {self.step.name: 1}
        else:
            value = self._arg_defaults()[self.step.name]
            if value <= args[self.root.max_name] - args[self.root.min_name] + 1:
                return {self.step.name: value}
            else:
                # Avoid OOB
                return {self.step.name: 1}
