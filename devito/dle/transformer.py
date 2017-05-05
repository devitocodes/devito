from collections import Sequence

from devito.dle.backends import (State, BasicRewriter, DevitoRewriter,
                                 DevitoSpeculativeRewriter, YaskRewriter)
from devito.logger import dle_warning


modes = {
    'basic': BasicRewriter,
    'advanced': DevitoRewriter,
    'speculative': DevitoSpeculativeRewriter,
    'yask': YaskRewriter
}


def transform(node, mode='basic', options=None, compiler=None):
    """
    Transform Iteration/Expression trees to generate highly optimized C code.

    :param node: The Iteration/Expression tree to be transformed, or an iterable
                 of Iteration/Expression trees.
    :param mode: Drive the tree transformation. ``mode`` is a string indicating
                 a certain optimization pipeline. The following values are accepted: ::

                     * 'noop': Do nothing
                     * 'basic': Add instructions to avoid denormal numbers and create
                                elemental functions for rapid JIT-compilation.
                     * 'advanced': 'basic', vectorization, loop blocking
                     * '3D-advanced': Like 'advanced', but apply 3D loop blocking
                                      if there are at least the perfectly nested
                                      parallel iteration spaces -- [S]
                     * 'speculative': Apply all of the 'advanced' transformations,
                                      plus other transformations that might increase
                                      (or possibly decrease) performance -- [S]
                     * 'yask': Optimize by offloading to the YASK optimizer.
    :param options: A dictionary with additional information to drive the DLE. The
                    following values are accepted: ::

                        * 'blockshape': A tuple representing the shape of a block created
                                        by loop blocking.
                        * 'blockinner': By default, loop blocking is not applied to the
                                        innermost dimension of an Iteration/Expression
                                        tree to maximize vectorization. Set this flag to
                                        True to override this heuristic.
                        * 'openmp': True to emit OpenMP code, False otherwise.
    :param compiler: Compiler class used to perform JIT compilation. Useful to
                     introduce compiler-specific vectorization pragmas.
    """
    assert mode is None or isinstance(mode, str)

    # Check input AST
    if isinstance(node, Sequence):
        assert all(n.is_Node for n in node)
        node = list(node)
    elif node.is_Node:
        node = [node]
    else:
        raise ValueError("Got illegal node of type %s." % type(node))

    # Check input options
    params = options.copy()
    for i in options:
        if i not in ('blockshape', 'blockinner', 'openmp'):
            dle_warning("Illegal DLE parameter '%s'" % i)
            params.pop(i)

    # Check input mode
    if mode is None or mode == 'noop':
        return State(node)
    elif mode == '3D-advanced':
        params['blockinner'] = True
        mode = 'advanced'
    elif mode not in modes:
        dle_warning("Unknown transformer mode(s) %s" % mode)
        return State(node)

    rewriter = modes[mode](node, params, compiler)
    return rewriter.run()
