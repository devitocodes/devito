
__all__ = ['Target']


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

    # Fetch the requested rewriter
    try:
        rewriter = targets.fetch(platform, mode)(params, platform)
    except KeyError:
        # Fallback: custom rewriter -- `mode` is a specific sequence of
        # transformation passes
        rewriter = targets.fetch(platform, 'custom')(mode, params, platform)

