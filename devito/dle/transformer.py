from collections import OrderedDict

from devito.archinfo import Platform, Cpu64
from devito.ir.iet import Node
from devito.dle.rewriters import CPU64Rewriter, CustomRewriter, State
from devito.exceptions import DLEException
from devito.logger import dle_warning
from devito.parameters import configuration

__all__ = ['modes', 'transform']


class DLEModes(OrderedDict):

    """
    The DLE transformation modes. This is a dictionary ``P -> {M -> R}``,
    where P is a Platform, M a rewrite mode (e.g., 'advanced', 'speculative'),
    and R a Rewriter.

    This dictionary is to be modified at backend-initialization time by adding
    all Platform-keyed mappers as supported by the specific backend.
    """

    def add(self, platform, mapper):
        assert issubclass(platform, Platform)
        assert isinstance(mapper, dict)
        assert all(i in mapper for i in ['advanced', 'speculative'])
        super(DLEModes, self).__setitem__(platform, mapper)

    def fetch(self, platform, mode):
        # Try to fetch the most specific Rewriter for the given Platform
        for cls in platform.__class__.mro():
            for k, v in self.items():
                if issubclass(k, cls):
                    return v[mode]
        raise KeyError("Couldn't find a rewriter mode for platform `%s`", platform)


modes = DLEModes()
modes.add(Cpu64, {'advanced': CPU64Rewriter, 'speculative': CPU64Rewriter})


def transform(iet, mode='advanced', options=None):
    """
    Transform Iteration/Expression trees (IET) to generate optimized C code.

    Parameters
    ----------
    iet : Node
        The root of the IET to be transformed.
    mode : str, optional
        The transformation mode.
        - ``noop``: Do nothing.
        - ``advanced``: flush denormals, vectorization, loop blocking, OpenMP-related
                        optimizations (collapsing, nested parallelism, ...), MPI-related
                        optimizations (aggregation of communications, reshuffling of
                        communications, ...).
        - ``speculative``: Apply all of the 'advanced' transformations, plus other
                           transformations that might increase (or possibly decrease)
                           performance.
    options : dict, optional
        - ``openmp``: Enable/disable OpenMP. Defaults to `configuration['openmp']`.
        - ``mpi``: Enable/disable MPI. Defaults to `configuration['mpi']`.
        - ``blockinner``: Enable/disable blocking of innermost loops. By default,
                          this is disabled to maximize SIMD vectorization. Pass True
                          to override this heuristic.
        - ``blockalways``: Pass True to unconditionally apply loop blocking, even when
                           the compiler heuristically thinks that it might not be
                           profitable and/or dangerous for performance.
    """
    assert isinstance(iet, Node)

    # Default options
    params = {}
    params['blockinner'] = configuration['dle-options'].get('blockinner', False)
    params['blockalways'] = configuration['dle-options'].get('blockalways', False)
    params['openmp'] = configuration['openmp']
    params['mpi'] = configuration['mpi']

    # Parse input options (potentially replacing defaults)
    for k, v in (options or {}).items():
        if k not in params:
            dle_warning("Illegal DLE option '%s'" % k)
        else:
            params[k] = v

    # Force OpenMP if parallelism was requested, even though mode is 'noop'
    if mode == 'noop' and params['openmp'] is True:
        mode = 'openmp'

    # What is the target platform for which the optimizations are applied?
    platform = configuration['platform']

    # Run the DLE
    if mode is None or mode == 'noop':
        # No-op case
        return iet, State(iet)
    try:
        # Preset rewriter (typical use case)
        rewriter = modes.fetch(platform, mode)(params, platform)
        return rewriter.run(iet)
    except KeyError:
        pass
    try:
        # Fallback: custom rewriter -- `mode` is a specific sequence of
        # transformation passes
        rewriter = CustomRewriter(mode, params, platform)
        return rewriter.run(iet)
    except DLEException:
        dle_warning("Unknown transformer mode(s) %s" % mode)
        return iet, State(iet)
