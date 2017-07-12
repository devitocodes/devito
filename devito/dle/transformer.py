from collections import Sequence

from devito.dle.backends import (State, BasicRewriter, DevitoCustomRewriter,
                                 DevitoRewriter, DevitoSpeculativeRewriter)
from devito.exceptions import DLEException
from devito.logger import dle_warning

__all__ = ['transform', 'modes']


modes = {
    'basic': BasicRewriter,
    'advanced': DevitoRewriter,
    'speculative': DevitoSpeculativeRewriter
}
"""The DLE transformation modes."""


def transform(node, mode='basic', options=None):
    """
    Transform Iteration/Expression trees to generate highly optimized C code.

    :param node: The Iteration/Expression tree to be transformed, or an iterable
                 of Iteration/Expression trees.
    :param mode: Drive the tree transformation. ``mode`` is a string indicating
                 a certain optimization pipeline. The following values are accepted: ::

                     * 'noop': Do nothing.
                     * 'basic': Add instructions to avoid denormal numbers and create
                                elemental functions for rapid JIT-compilation.
                     * 'advanced': 'basic', vectorization, loop blocking.
                     * '3D-advanced': Like 'advanced', but attempt 3D loop blocking.
                     * 'speculative': Apply all of the 'advanced' transformations,
                                      plus other transformations that might increase
                                      (or possibly decrease) performance.
    :param options: A dictionary with additional information to drive the DLE. The
                    following values are accepted: ::

                        * 'blockshape': A tuple representing the shape of a block created
                                        by loop blocking.
                        * 'blockinner': By default, loop blocking is not applied to the
                                        innermost dimension of an Iteration/Expression
                                        tree to maximize vectorization. Set this flag to
                                        True to override this heuristic.
    """
    from devito.parameters import configuration

    # Check input parameters
    if not (mode is None or isinstance(mode, str)):
        raise ValueError("Parameter 'mode' should be a string, not %s." % type(mode))

    if isinstance(node, Sequence):
        assert all(n.is_Node for n in node)
        node = list(node)
    elif node.is_Node:
        node = [node]
    else:
        raise ValueError("Got illegal node of type %s." % type(node))

    # Parse options
    options = options or {}
    params = options.copy()
    for i in options:
        if i not in ('blockshape', 'blockinner'):
            dle_warning("Illegal DLE parameter '%s'" % i)
            params.pop(i)
    params['compiler'] = configuration['compiler']
    params['openmp'] = configuration['openmp']
    if mode == '3D-advanced':
        params['blockinner'] = True
        mode = 'advanced'

    # Process the Iteration/Expression tree through the DLE
    if mode is None or mode == 'noop':
        return State(node)
    elif mode not in modes:
        try:
            rewriter = DevitoCustomRewriter(node, mode, params)
            return rewriter.run()
        except DLEException:
            dle_warning("Unknown transformer mode(s) %s" % mode)
            return State(node)
    else:
        rewriter = modes[mode](node, params)
        return rewriter.run()
