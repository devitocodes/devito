from collections import OrderedDict

from devito.archinfo import Platform, Cpu64
from devito.ir.iet import Node
from devito.dle.rewriters import CPU64Rewriter, CustomRewriter, PlatformRewriter, State
from devito.logger import dle as log, dle_warning as warning
from devito.parameters import configuration

__all__ = ['dle_registry', 'modes', 'transform']


dle_registry = ('noop', 'advanced', 'speculative')


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
        assert all(i in mapper for i in dle_registry)
        super(DLEModes, self).__setitem__(platform, mapper)

    def fetch(self, platform, mode):
        # Try to fetch the most specific Rewriter for the given Platform
        for cls in platform.__class__.mro():
            for k, v in self.items():
                if issubclass(k, cls):
                    return v[mode]
        raise KeyError("Couldn't find a rewriter mode for platform `%s`", platform)


modes = DLEModes()
modes.add(Cpu64, {'noop': PlatformRewriter,
                  'advanced': CPU64Rewriter,
                  'speculative': CPU64Rewriter})


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
        - ``blocklevels``: Levels of blocking for hierarchical tiling (blocks,
                           sub-blocks, sub-sub-blocks, ...). Different Platforms have
                           different default values.
    """
    assert isinstance(iet, Node)

    # Default options
    params = {}
    params['blockinner'] = configuration['dle-options'].get('blockinner', False)
    params['blockalways'] = configuration['dle-options'].get('blockalways', False)
    params['blocklevels'] = configuration['dle-options'].get('blocklevels', None)
    params['openmp'] = configuration['openmp']
    params['mpi'] = configuration['mpi']

    # Parse input options (potentially replacing defaults)
    for k, v in (options or {}).items():
        if k not in params:
            warning("Illegal DLE option '%s'" % k)
        else:
            params[k] = v

    # Force OpenMP/MPI if parallelism was requested, even though mode is 'noop'
    # FIXME: Possibly nonsense?
    if mode == 'noop' and any([i for i in [params['openmp'], params['mpi']]]):
        mode = 'advanced'

    # What is the target platform for which the optimizations are applied?
    platform = configuration['platform']

    # No-op case
    if mode is None or mode == 'noop':
        return iet, State(iet)

    # Fetch the requested rewriter
    try:
        rewriter = modes.fetch(platform, mode)(params, platform)
    except KeyError:
        # Fallback: custom rewriter -- `mode` is a specific sequence of
        # transformation passes
        rewriter = CustomRewriter(mode, params, platform)

    # Trigger the DLE passes
    state = rewriter.run(iet)

    # Print out the profiling data
    print_profiling(state)

    return state.root, state


def print_profiling(state):
    """
    Print a summary of the applied transformations.
    """
    timings = state.timings

    if configuration['profiling'] in ['basic', 'advanced']:
        row = "%s (elapsed: %.2f s)"
        out = "\n     ".join(row % ("".join(filter(lambda c: not c.isdigit(), k[1:])), v)
                             for k, v in timings.items())
        elapsed = sum(timings.values())
        log("%s\n     [Total elapsed: %.2f s]" % (out, elapsed))
    else:
        # Shorter summary
        log("passes: %s (elapsed %.2f s)" % (",".join(i[1:] for i in timings),
                                             sum(timings.values())))
