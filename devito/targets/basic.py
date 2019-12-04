from collections import OrderedDict

from devito.archinfo import Platform
from devito.ir.iet import Node
from devito.logger import dle as log, dle_warning as warning
from devito.parameters import configuration
from devito.targets.common import Graph
from devito.tools import Singleton

__all__ = ['dle_registry', 'iet_lower', 'targets', 'Target']


#TODO: change this
dle_registry = ('noop', 'advanced')


class Target(object):

    """
    Specialize an Iteration/Expression tree (IET) introducing Target-specific
    optimizations, parallelism, etc.
    """

    def __init__(self, params, platform):
        self.params = params
        self.platform = platform

    def process(self, iet):
        graph = Graph(iet)

        self._pipeline(graph)

        return graph

    def _pipeline(self, graph):
        """The rewrite passes."""
        return


class TargetsMap(OrderedDict, metaclass=Singleton):

    """
    A special mapper for Targets:

        (platform, mode, language) -> target

    where:

        * `platform` is an object of type Platform, that is the architecture
          the generated code should be specializer for.
        * `mode` is the optimization level (e.g., `advanced`)
        * `language` is the generated code language (default is C+OpenMP+MPI,
          but in the future it could also be OpenACC or CUDA.
        * `target` is an object of type Target.
    """

    def add(self, target, platform, mode, language='C'):
        assert issubclass(target, Target)
        assert issubclass(platform, Platform)
        assert mode in dle_registry or mode == 'custom'

        self[(platform, mode, language)] = target

    def fetch(self, platform, mode, language='C'):
        # Retrieve the most specialized Target
        for cls in platform.__class__.mro():
            for (p, m, l), target in self.items():
                if issubclass(p, cls) and m == mode and l == language:
                    return target
        raise KeyError("Couldn't find a Target for `%s`" % str((p, m, l)))


targets = TargetsMap()
"""To be populated by the individual backends."""


def iet_lower(iet, mode='advanced', options=None):
    """
    Specialize an Iteration/Expression tree (IET) for a certain Target. This
    consists of:

        * Loop-level transformations to generate Target-optimized code;
        * Shared-memory and distributed-memory parallelism;
        * Symbol and data management.

    Parameters
    ----------
    iet : Node
        The IET to be lowered.
    mode : str, optional
        The DLE transformation mode.
        - ``noop``: Do nothing.
        - ``advanced``: flush denormals, vectorization, loop blocking, OpenMP-related
                        optimizations (collapsing, nested parallelism, ...), MPI-related
                        optimizations (aggregation of communications, reshuffling of
                        communications, ...).
    options : dict, optional
        - ``openmp``: Enable/disable OpenMP. Defaults to `configuration['openmp']`.
        - ``mpi``: Enable/disable MPI. Defaults to `configuration['mpi']`.
        - ``blockinner``: Enable/disable blocking of innermost loops. By default,
                          this is disabled to maximize SIMD vectorization. Pass True
                          to override this heuristic.
        - ``blocklevels``: Levels of blocking for hierarchical tiling (blocks,
                           sub-blocks, sub-sub-blocks, ...). Different Platforms have
                           different default values.
        - ``language``: To generate code using a low-level language other than the
                        default one. Currently this option is ignored, but in the
                        future we expect to have multiple alternatives for the same
                        platform. For example, if the platform in question is NVIDIAX,
                        one may want to generate OpenMP or OpenACC offloading code --
                        ``language=openmp`` and ``language=openacc`` respectively) --
                        rather than the default ``language=cuda``.
    """
    assert isinstance(iet, Node)

    # What is the target platform for which the optimizations are applied?
    platform = configuration['platform']

    # Default options
    params = {}
    params['blockinner'] = configuration['dle-options'].get('blockinner', False)
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
    if mode is None:
        mode = 'noop'
    elif mode == 'noop':
        mode = tuple(i for i in ['mpi', 'openmp'] if params[i]) or 'noop'

    # Fetch the requested rewriter
    try:
        rewriter = targets.fetch(platform, mode)(params, platform)
    except KeyError:
        # Fallback: custom rewriter -- `mode` is a specific sequence of
        # transformation passes
        rewriter = targets.fetch(platform, 'custom')(mode, params, platform)

    # Trigger the DLE passes
    graph = rewriter.process(iet)

    # Print out the profiling data
    print_profiling(graph)

    return graph.root, graph


def print_profiling(graph):
    """
    Print a summary of the applied transformations.
    """
    timings = graph.timings

    if configuration['profiling'] in ['basic', 'advanced']:
        row = "%s (elapsed: %.2f s)"
        out = "\n     ".join(row % ("".join(filter(lambda c: not c.isdigit(), k)), v)
                             for k, v in timings.items())
        elapsed = sum(timings.values())
        log("%s\n     [Total elapsed: %.2f s]" % (out, elapsed))
    else:
        # Shorter summary
        log("passes: %s (elapsed %.2f s)" % (",".join(timings), sum(timings.values())))
