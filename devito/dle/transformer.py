from devito.ir.iet import Node
from devito.dle.backends import State, CustomRewriter
from devito.exceptions import DLEException
from devito.logger import dle_warning
from devito.parameters import configuration

__all__ = ['init_dle', 'transform']


default_modes = {
    'basic': None,
    'advanced': None,
    'advanced-safemath': None,
    'speculative': None
}
"""The DLE transformation modes.
This dictionary may be modified at backend-initialization time."""

default_options = {
    'blockinner': False,
    'blockalways': False
}
"""Default values for the supported optimization options.
This dictionary may be modified at backend-initialization time."""

configuration.add('dle', 'advanced', list(default_modes))
configuration.add('dle-options',
                  ';'.join('%s:%s' % (k, v) for k, v in default_options.items()),
                  list(default_options))

all_options = tuple(default_options) + ('openmp',)


def init_dle(backend_modes):
    global default_modes
    for i in list(default_modes):
        default_modes[i] = backend_modes[i]


def transform(iet, mode='basic', options=None):
    """
    Transform Iteration/Expression trees (IET) to generate optimized C code.

    Parameters
    ----------
    iet : Node
        The root of the IET to be transformed.
    mode : str, optional
        The transformation mode.
        * 'noop': Do nothing.
        * 'basic': Add instructions to avoid denormal numbers and create elemental
                   functions for quicker JIT-compilation.
        * 'advanced': 'basic', vectorization, loop blocking.
        * 'speculative': Apply all of the 'advanced' transformations, plus other
                         transformations that might increase (or possibly decrease)
                         performance.
    options : dict, optional
        * 'openmp': Enable/disable OpenMP. Defaults to `configuration['openmp']`.
        * 'blockinner': Enable/disable blocking of innermost loops. By default,
                        this is disabled to maximize SIMD vectorization. Pass True
                        to override this heuristic.
        * 'blockalways': Pass True to unconditionally apply loop blocking, even when
                         the compiler heuristically thinks that it might not be
                         profitable and/or dangerous for performance.
    """
    assert isinstance(iet, Node)

    # Parse options (local values take precedence over global ones)
    options = options or {}
    params = options.copy()
    for i in options:
        if i not in all_options:
            dle_warning("Illegal DLE option '%s'" % i)
            params.pop(i)
    params.update({k: v for k, v in configuration['dle-options'].items()
                   if k not in params})
    params.setdefault('openmp', configuration['openmp'])

    # Force OpenMP if parallelism was requested, even though mode is 'noop'
    if mode == 'noop' and params['openmp'] is True:
        mode = 'openmp'

    # Process the Iteration/Expression tree through the DLE
    if mode is None or mode == 'noop':
        return State(iet)
    elif mode not in default_modes:
        try:
            rewriter = CustomRewriter(iet, mode, params)
            return rewriter.run()
        except DLEException:
            dle_warning("Unknown transformer mode(s) %s" % mode)
            return State(iet)
    else:
        rewriter = default_modes[mode](iet, params)
        return rewriter.run()
