from devito.core.autotuning import autotune
from devito.exceptions import InvalidOperator
from devito.logger import warning
from devito.parameters import configuration
from devito.operator import Operator
from devito.tools import as_tuple, timed_pass
from devito.types import NThreads

__all__ = ['CoreOperator', 'CustomOperator']


class BasicOperator(Operator):

    # The various Operator subclasses are expected to override the following attributes

    _Parallelizer = None
    """
    Used to generate shared-memory parallel code by various compiler passes.
    """

    _DataManager = None
    """
    Used to place symbol definitions as well as data allocation, movement, and deletion.
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        # Will be populated with dummy values; this method is actually overriden
        # by the subclasses
        o = {}
        oo = kwargs['options']

        # Execution modes
        o['mpi'] = False
        o['parallel'] = False

        if oo:
            raise InvalidOperator("Unrecognized optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs

    def _autotune(self, args, setup):
        if setup in [False, 'off']:
            return args
        elif setup is True:
            level, mode = configuration['autotuning']
            level = level or 'basic'
            args, summary = autotune(self, args, level, mode)
        elif isinstance(setup, str):
            _, mode = configuration['autotuning']
            args, summary = autotune(self, args, setup, mode)
        elif isinstance(setup, tuple) and len(setup) == 2:
            level, mode = setup
            if level is False:
                return args
            else:
                args, summary = autotune(self, args, level, mode)
        else:
            raise ValueError("Expected bool, str, or 2-tuple, got `%s` instead"
                             % type(setup))

        # Record the tuned values
        self._state.setdefault('autotuning', []).append(summary)

        return args

    @property
    def nthreads(self):
        nthreads = [i for i in self.input if type(i).__base__ is NThreads]
        if len(nthreads) == 0:
            return 1
        else:
            assert len(nthreads) == 1
            return nthreads.pop()


class CoreOperator(BasicOperator):
    pass


class CustomOperator(BasicOperator):

    @classmethod
    def _make_dsl_passes_mapper(cls, **kwargs):
        return {}

    @classmethod
    def _make_exprs_passes_mapper(cls, **kwargs):
        return {}

    @classmethod
    def _make_clusters_passes_mapper(cls, **kwargs):
        return {}

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        # Dummy values
        noop = lambda i: i
        return {
            'mpi': noop,
            'parallel': noop
        }

    _known_passes = ()
    _known_passes_disabled = ()

    @classmethod
    def _build(cls, expressions, **kwargs):
        # Sanity check
        passes = as_tuple(kwargs['mode'])
        for i in passes:
            if i not in cls._known_passes:
                if i in cls._known_passes_disabled:
                    warning("Got explicit pass `%s`, but it's unsupported on an "
                            "Operator of type `%s`" % (i, str(cls)))
                else:
                    raise InvalidOperator("Unknown pass `%s`" % i)

        return super()._build(expressions, **kwargs)

    @classmethod
    @timed_pass(name='specializing.DSL')
    def _specialize_dsl(cls, expressions, **kwargs):
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_dsl_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                expressions = passes_mapper[i](expressions)
            except KeyError:
                pass

        return expressions

    @classmethod
    @timed_pass(name='specializing.Expressions')
    def _specialize_exprs(cls, expressions, **kwargs):
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_exprs_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                expressions = passes_mapper[i](expressions)
            except KeyError:
                pass

        return expressions

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_clusters_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                clusters = passes_mapper[i](clusters)
            except KeyError:
                pass

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']
        passes = as_tuple(kwargs['mode'])

        # Run passes
        passes_mapper = cls._make_iet_passes_mapper(**kwargs)
        applied = []
        for i in passes:
            try:
                applied.append(passes_mapper[i])
                passes_mapper[i](graph)
            except KeyError:
                pass

        # Force-call `mpi` if requested via global option
        if passes_mapper['mpi'] not in applied and options['mpi']:
            passes_mapper['mpi'](graph)

        # Parallelism
        if passes_mapper['parallel'] not in applied and options['parallel']:
            passes_mapper['parallel'](graph)

        # Symbol definitions
        cls._DataManager(cls._Parallelizer, sregistry, options).process(graph)

        # Initialize the target-language runtime
        if passes_mapper['init'] not in applied:
            passes_mapper['init'](graph)

        return graph
