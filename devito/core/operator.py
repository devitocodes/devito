from collections.abc import Iterable
from functools import cached_property

import numpy as np

from devito.core.autotuning import autotune
from devito.exceptions import InvalidOperator
from devito.ir import FindSymbols
from devito.logger import warning
from devito.mpi.routines import mpi_registry
from devito.parameters import configuration
from devito.operator import Operator
from devito.tools import (as_tuple, is_integer, timed_pass,
                          UnboundTuple, UnboundedMultiTuple)
from devito.types import NThreads, PThreadArray

__all__ = ['CoreOperator', 'CustomOperator',
           # Optimization options
           'ParTile']


class BasicOperator(Operator):

    # Default values for various optimization options

    CSE_MIN_COST = 1
    """
    Minimum computational cost of an operation to be eliminated as a
    common sub=expression.
    """

    CSE_ALGO = 'basic'
    """
    The algorithm to use for common sub-expression elimination.
    """

    FACT_SCHEDULE = 'basic'
    """
    The schedule to use for the computation of factorizations.
    """

    BLOCK_LEVELS = 1
    """
    Loop blocking depth. So, 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    BLOCK_EAGER = True
    """
    Apply loop blocking as early as possible, and in particular prior to CIRE.
    """

    BLOCK_RELAX = False
    """
    If set to True, bypass the compiler heuristics that prevent loop blocking in
    situations where the performance impact might be detrimental.
    """

    CIRE_MINGAIN = 10
    """
    Minimum operation count reduction for a redundant expression to be optimized
    away. Higher (lower) values make a redundant expression less (more) likely to
    be optimized away.
    """

    CIRE_SCHEDULE = 'automatic'
    """
    Strategy used to schedule derivatives across loops. This impacts the operational
    intensity of the generated kernel.
    """

    SCALAR_MIN_TYPE = np.float16
    """
    Minimum datatype for a scalar arising from a common sub-expression or CIRE temp.
    """

    PAR_COLLAPSE_NCORES = 4
    """
    Use a collapse clause if the number of available physical cores is greater
    than this threshold.
    """

    PAR_COLLAPSE_WORK = 100
    """
    Use a collapse clause if the trip count of the collapsable loops is statically
    known to exceed this threshold.
    """

    PAR_CHUNK_NONAFFINE = 3
    """
    Coefficient to adjust the chunk size in non-affine parallel loops.
    """

    PAR_DYNAMIC_WORK = 10
    """
    Use dynamic scheduling if the operation count per iteration exceeds this
    threshold. Otherwise, use static scheduling.
    """

    PAR_NESTED = 2
    """
    Use nested parallelism if the number of hyperthreads per core is greater
    than this threshold.
    """

    MAPIFY_REDUCE = False
    """
    Vector-expand all scalar reductions to turn them into explicit map-reductions,
    which may be easier to parallelize for certain backends.
    """

    EXPAND = True
    """
    Unroll all loops with short, numeric trip count, such as loops created by
    finite-difference derivatives.
    """

    DERIV_SCHEDULE = 'basic'
    """
    The schedule to use for the computation of finite-difference derivatives.
    Only meaningful when `EXPAND=False`.
    """

    MPI_MODES = tuple(mpi_registry)
    """
    The supported MPI modes.
    """

    DIST_DROP_UNWRITTEN = True
    """
    Drop halo exchanges for read-only Function, even in presence of
    stencil-like data accesses.
    """

    INDEX_MODE = "int32"
    """
    The type of the expression used to compute array indices. Either `int32`
    (default) or `int64`.
    """

    LINEARIZE = False
    """
    Linearize n-dimensional Indexeds.
    """

    ERRCTL = None
    """
    Runtime error checking. If this option is enabled, the generated code will
    include runtime checks for various things that might go south, such as
    instability (e.g., NaNs), failed library calls (e.g., kernel launches).
    """

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

    @classmethod
    def _check_kwargs(cls, **kwargs):
        oo = kwargs['options']

        if oo['mpi'] and oo['mpi'] not in cls.MPI_MODES:
            raise InvalidOperator("Unsupported MPI mode `%s`" % oo['mpi'])

        if oo['cse-algo'] not in ('basic', 'smartsort', 'advanced'):
            raise InvalidOperator("Illegal `cse-algo` value")

        if oo['deriv-schedule'] not in ('basic', 'smart'):
            raise InvalidOperator("Illegal `deriv-schedule` value")
        if oo['deriv-unroll'] not in (False, 'inner', 'full'):
            raise InvalidOperator("Illegal `deriv-unroll` value")

        if oo['errctl'] not in (None, False, 'basic', 'max'):
            raise InvalidOperator("Illegal `errctl` value")

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

    @cached_property
    def nthreads(self):
        nthreads = [i for i in self.input if isinstance(i, NThreads)]
        if len(nthreads) == 0:
            return 1
        else:
            assert len(nthreads) == 1
            return nthreads.pop()

    @cached_property
    def npthreads(self):
        symbols = FindSymbols().visit(self.body)
        ptas = [i for i in symbols if isinstance(i, PThreadArray)]
        return sum(i.size for i in ptas)


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
            passes_mapper['mpi'](graph, **kwargs)

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

        # Symbol definitions
        cls._Target.DataManager(**kwargs).process(graph)

        # Linearize n-dimensional Indexeds
        if 'linearize' not in passes and options['linearize']:
            passes_mapper['linearize'](graph)

        # Enforce pthreads if CPU-GPU orchestration requested
        if 'orchestrate' in passes and 'pthreadify' not in passes:
            passes_mapper['pthreadify'](graph, sregistry=sregistry)

        return graph


# Wrappers for optimization options


class OptOption:
    pass


class ParTileArg(UnboundTuple):

    def __new__(cls, items, rule=None, tag=None):
        if items is None:
            items = tuple()
        obj = super().__new__(cls, *items)
        obj.rule = rule
        obj.tag = tag
        return obj


class ParTile(UnboundedMultiTuple, OptOption):

    def __new__(cls, items, default=None, sparse=None, reduce=None):
        if not items:
            return UnboundedMultiTuple()
        elif isinstance(items, bool):
            if not default:
                raise ValueError("Expected `default` value, got None")
            items = (ParTileArg(as_tuple(default)),)
        elif isinstance(items, (list, tuple)):
            if not items:
                raise ValueError("Expected at least one value")

            # Normalize to tuple of ParTileArgs

            x = items[0]
            if is_integer(x):
                # E.g., 32
                items = (ParTileArg(items),)

            elif x is None:
                # E.g. (None, None); to define the dimensionality of a block,
                # while the actual shape values remain parametric
                items = (ParTileArg(items),)

            elif isinstance(x, ParTileArg):
                # From a reconstruction
                pass

            elif isinstance(x, Iterable):
                if not x:
                    raise ValueError("Expected at least one value")

                try:
                    y = items[1]
                    if is_integer(y) or isinstance(y, str) or y is None:
                        # E.g., ((32, 4, 8), 'rule')
                        # E.g., ((32, 4, 8), 'rule', 'tag')
                        items = (ParTileArg(*items),)
                    else:
                        try:
                            # E.g., (((32, 4, 8), 'rule'), ((32, 4, 4), 'rule'))
                            # E.g., (((32, 4, 8), 'rule0', 'tag0'),
                            #        ((32, 4, 4), 'rule1', 'tag1'))
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
            raise ValueError("Expected bool or iterable, got %s instead" % type(items))

        obj = super().__new__(cls, *items)
        obj.default = as_tuple(default)
        obj.sparse = as_tuple(sparse)
        obj.reduce = as_tuple(reduce)

        return obj

    @property
    def is_multi(self):
        return len(self) > 1
