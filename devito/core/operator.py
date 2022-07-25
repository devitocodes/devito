from collections.abc import Iterable

from devito.core.autotuning import autotune
from devito.exceptions import InvalidOperator
from devito.logger import warning
from devito.parameters import configuration
from devito.operator import Operator
from devito.tools import as_tuple, is_integer, timed_pass
from devito.types import NThreads

__all__ = ['CoreOperator', 'CustomOperator',
           # Optimization options
           'ParTile']


class BasicOperator(Operator):

    # The various Operator subclasses are expected to override the following attributes

    _Target = None
    """
    The target language constructor, to be specified by subclasses.
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
        nthreads = [i for i in self.input if isinstance(i, NThreads)]
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

        passes_mapper = cls._make_iet_passes_mapper(**kwargs)

        # Always attempt `mpi` codegen before anything else to maximize the
        # outcome of the other passes (e.g., shared-memory parallelism benefits
        # from HaloSpot optimization)
        # Note that if MPI is disabled then this pass will act as a no-op
        if 'mpi' not in passes:
            passes_mapper['mpi'](graph)

        # Run passes
        applied = []
        for i in passes:
            try:
                applied.append(passes_mapper[i])
                passes_mapper[i](graph)
            except KeyError:
                pass

        # Parallelism
        if passes_mapper['parallel'] not in applied and options['parallel']:
            passes_mapper['parallel'](graph)

        # Initialize the target-language runtime
        if 'init' not in passes:
            passes_mapper['init'](graph)

        # Enforce pthreads if CPU-GPU orchestration requested
        if 'orchestrate' in passes and 'pthreadify' not in passes:
            passes_mapper['pthreadify'](graph, sregistry=sregistry)

        # Symbol definitions
        cls._Target.DataManager(sregistry, options).process(graph)

        # Linearize n-dimensional Indexeds
        if 'linearize' not in passes and options['linearize']:
            passes_mapper['linearize'](graph)

        return graph


# Wrappers for optimization options


class OptOption(object):
    pass


class ParTileArg(tuple):

    def __new__(cls, items, shm=0, tag=None):
        obj = super().__new__(cls, items)
        obj.shm = shm
        obj.tag = tag
        return obj


class ParTile(tuple, OptOption):

    def __new__(cls, items, default=None):
        if not items:
            return None
        elif isinstance(items, bool):
            if not default:
                raise ValueError("Expected `default` value, got None")
            items = (ParTileArg(as_tuple(default)),)
        elif isinstance(items, tuple):
            if not items:
                raise ValueError("Expected at least one value")

            # Normalize to tuple of ParTileArgs

            x = items[0]
            if is_integer(x):
                # E.g., (32, 4, 8)
                items = (ParTileArg(items),)

            elif isinstance(x, Iterable):
                if not x:
                    raise ValueError("Expected at least one value")

                try:
                    y = items[1]
                    if is_integer(y):
                        # E.g., ((32, 4, 8), 1)
                        # E.g., ((32, 4, 8), 1, 'tag')
                        items = (ParTileArg(*items),)
                    else:
                        try:
                            # E.g., (((32, 4, 8), 1), ((32, 4, 4), 2))
                            # E.g., (((32, 4, 8), 1, 'tag0'), ((32, 4, 4), 2, 'tag1'))
                            items = tuple(ParTileArg(*i) for i in items)
                        except TypeError:
                            # E.g., ((32, 4, 8), (32, 4, 4))
                            items = tuple(ParTileArg(i) for i in items)
                except IndexError:
                    # E.g., ((32, 4, 8),)
                    items = (ParTileArg(x),)
            else:
                raise ValueError("Expected int or tuple, got %s instead" % type(x))
        else:
            raise ValueError("Expected bool or tuple, got %s instead" % type(items))

        return super().__new__(cls, items)
