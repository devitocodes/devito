from itertools import groupby, product

import cgen as c
import numpy as np
from cached_property import cached_property

from devito.ir.iet import (Expression, Iteration, List, FindAdjacent, FindNodes,
                           IsPerfectIteration, Transformer, PARALLEL, AFFINE, make_efunc,
                           compose_nodes, filter_iterations, retrieve_iteration_tree)
from devito.exceptions import InvalidArgument
from devito.symbolics import as_symbol, xreplace_indices
from devito.tools import all_equal, as_tuple, flatten
from devito.types import IncrDimension, Scalar

__all__ = ['Blocker', 'BlockDimension']


class Blocker(object):

    def __init__(self, blockinner, blockalways, nlevels):
        self.blockinner = bool(blockinner)
        self.blockalways = bool(blockalways)
        self.nlevels = nlevels

        self.nblocked = 0

    def make_blocking(self, iet):
        """
        Apply loop blocking to PARALLEL Iteration trees.
        """
        # Make sure loop blocking will span as many Iterations as possible
        iet = fold_blockable_tree(iet, self.blockinner)

        mapper = {}
        efuncs = []
        block_dims = []
        for tree in retrieve_iteration_tree(iet):
            # Is the Iteration tree blockable ?
            iterations = filter_iterations(tree, lambda i: i.is_Parallel and i.is_Affine)
            if not self.blockinner:
                iterations = iterations[:-1]
            if len(iterations) <= 1:
                continue
            root = iterations[0]
            if not self.blockalways:
                # Heuristically bypass loop blocking if we think `tree`
                # won't be computationally expensive. This will help with code
                # size/readbility, JIT time, and auto-tuning time
                if not (tree.root.is_Sequential or iet.is_Callable):
                    # E.g., not inside a time-stepping Iteration
                    continue
                if any(i.dim.is_Sub and i.dim.local for i in tree):
                    # At least an outer Iteration is over a local SubDimension,
                    # which suggests the computational cost of this Iteration
                    # nest will be negligible w.r.t. the "core" Iteration nest
                    # (making use of non-local (Sub)Dimensions only)
                    continue
            if not IsPerfectIteration().visit(root):
                # Don't know how to block non-perfect nests
                continue

            # Apply hierarchical loop blocking to `tree`
            level_0 = []  # Outermost level of blocking
            level_i = [[] for i in range(1, self.nlevels)]  # Inner levels of blocking
            intra = []  # Within the smallest block
            for i in iterations:
                template = "%s%d_blk%s" % (i.dim.name, self.nblocked, '%d')
                properties = (PARALLEL,) + ((AFFINE,) if i.is_Affine else ())

                # Build Iteration across `level_0` blocks
                d = BlockDimension(i.dim, name=template % 0)
                level_0.append(Iteration([], d, d.symbolic_max, properties=properties))

                # Build Iteration across all `level_i` blocks, `i` in (1, self.nlevels]
                for n, li in enumerate(level_i, 1):
                    di = BlockDimension(d, name=template % n)
                    li.append(Iteration([], di, limits=(d, d+d.step-1, di.step),
                                        properties=properties))
                    d = di

                # Build Iteration within the smallest block
                intra.append(i._rebuild([], limits=(d, d+d.step-1, 1), offsets=(0, 0)))
            level_i = flatten(level_i)

            # Track all constructed BlockDimensions
            block_dims.extend(i.dim for i in level_0 + level_i)

            # Construct the blocked tree
            blocked = compose_nodes(level_0 + level_i + intra + [iterations[-1].nodes])
            blocked = unfold_blocked_tree(blocked)

            # Promote to a separate Callable
            dynamic_parameters = flatten((l0.dim, l0.step) for l0 in level_0)
            dynamic_parameters.extend([li.step for li in level_i])
            efunc = make_efunc("bf%d" % self.nblocked, blocked, dynamic_parameters)
            efuncs.append(efunc)

            # Compute the iteration ranges
            ranges = []
            for i, l0 in zip(iterations, level_0):
                maxb = i.symbolic_max - (i.symbolic_size % l0.step)
                ranges.append(((i.symbolic_min, maxb, l0.step),
                               (maxb + 1, i.symbolic_max, i.symbolic_max - maxb)))

            # Build Calls to the `efunc`
            body = []
            for p in product(*ranges):
                dynamic_args_mapper = {}
                for l0, (m, M, b) in zip(level_0, p):
                    dynamic_args_mapper[l0.dim] = (m, M)
                    dynamic_args_mapper[l0.step] = (b,)
                    for li in level_i:
                        if li.dim.root is l0.dim.root:
                            value = li.step if b is l0.step else b
                            dynamic_args_mapper[li.step] = (value,)
                call = efunc.make_call(dynamic_args_mapper)
                body.append(List(body=call))

            mapper[root] = List(body=body)

            # Next blockable nest, use different (unique) variable/function names
            self.nblocked += 1

        iet = Transformer(mapper).visit(iet)

        # Force-unfold if some folded Iterations haven't been blocked in the end
        iet = unfold_blocked_tree(iet)

        return iet, {'dimensions': block_dims,
                     'efuncs': efuncs,
                     'args': [i.step for i in block_dims]}


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
    if not all({PARALLEL, AFFINE}.issubset(set(i.properties)) for i in nodes):
        return False
    return (all_equal(i.dim for i in nodes) and
            all_equal(i.limits for i in nodes) and
            all_equal(i.index for i in nodes))


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

        # We can optimize the folded trees only iff:
        # test0 := they compute temporary arrays, but not if they compute input data
        # test1 := the outer Iterations have actually been blocked
        exprs = FindNodes(Expression).visit(tree)
        writes = [j.write for j in exprs if j.is_tensor]
        test0 = not all(j.is_Array for j in writes)
        test1 = any(not isinstance(j.limits[0], BlockDimension) for j in root)
        if test0 or test1:
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
        replaced = xreplace_indices([j.expr for j in exprs], mapper,
                                    lambda i: i.function not in writes, True)
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

    is_PerfKnob = True

    @cached_property
    def symbolic_min(self):
        return Scalar(name=self.min_name, dtype=np.int32, is_const=True)

    @property
    def _arg_names(self):
        return (self.step.name,)

    def _arg_defaults(self, **kwargs):
        # TODO: need a heuristic to pick a default block size
        return {self.step.name: 8}

    def _arg_values(self, args, interval, grid, **kwargs):
        if self.step.name in kwargs:
            return {self.step.name: kwargs.pop(self.step.name)}
        elif isinstance(self.parent, BlockDimension):
            # `self` is a BlockDimension within an outer BlockDimension, but
            # no value supplied -> the sub-block will span the entire block
            return {self.step.name: args[self.parent.step.name]}
        else:
            value = self._arg_defaults()[self.step.name]
            if value <= args[self.root.max_name] - args[self.root.min_name] + 1:
                return {self.step.name: value}
            else:
                # Avoid OOB (will end up here only in case of tiny iteration spaces)
                return {self.step.name: 1}

    def _arg_check(self, args, interval):
        """Check the block size won't cause OOB accesses."""
        value = args[self.step.name]
        if isinstance(self.parent, BlockDimension):
            # sub-BlockDimensions must be perfect divisors of their parent
            parent_value = args[self.parent.step.name]
            if parent_value % value > 0:
                raise InvalidArgument("Illegal block size `%s=%d`: sub-block sizes "
                                      "must divide the parent block size evenly (`%s=%d`)"
                                      % (self.step.name, value,
                                         self.parent.step.name, parent_value))
        else:
            if value < 0:
                raise InvalidArgument("Illegal block size `%s=%d`: it should be > 0"
                                      % (self.step.name, value))
            if value > args[self.root.max_name] - args[self.root.min_name] + 1:
                # Avoid OOB
                raise InvalidArgument("Illegal block size `%s=%d`: it's greater than the "
                                      "iteration range and it will cause an OOB access"
                                      % (self.step.name, value))
