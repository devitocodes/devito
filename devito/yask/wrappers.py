import ctypes
import os
import warnings
from importlib import import_module, invalidate_caches
from glob import glob
from subprocess import call
from collections import OrderedDict

from codepy.jit import CacheLockManager, CleanupManager

from devito.compiler import make
from devito.logger import debug, yask as log, yask_warning as warning
from devito.tools import Signer, powerset, filter_sorted

from devito.yask import cfac, ofac, exit, configuration
from devito.yask.utils import namespace
from devito.yask.transformer import make_yask_ast


class YaskKernel(object):

    """
    A ``YaskKernel`` wraps a YASK kernel solution.
    """

    def __init__(self, name, yc_soln, local_grids=None):
        """
        Write out a YASK kernel, build it using YASK's Makefiles,
        import the corresponding SWIG-generated Python module, and finally
        create a YASK kernel solution object.

        :param name: Unique name of this YaskKernel.
        :param yc_soln: YaskCompiler solution.
        :param local_grids: A local grid is necessary to run the YaskKernel,
                            but its final content can be ditched. Indeed, local
                            grids are hidden to users -- for example, they could
                            represent temporary arrays introduced by the DSE.
                            This parameter tells which of the ``yc_soln``'s grids
                            are local.
        """
        self.name = name

        # Shared object name
        self.soname = "%s.devito.%s" % (name, configuration['platform'])

        if os.path.exists(os.path.join(namespace['yask-pylib'], '%s.py' % name)):
            # Nothing to do -- the YASK solution was compiled in a previous session
            yk = import_module(name)
            debug("cache hit, `%s` imported w/o jitting" % name)
        else:
            # We create and JIT compile a fresh YASK solution

            # The lock manager prevents race conditions
            # `lock_m` is used only to keep the lock manager alive
            with warnings.catch_warnings():
                cleanup_m = CleanupManager()
                lock_m = CacheLockManager(cleanup_m, namespace['yask-output-dir'])  # noqa

            # The directory in which the YASK-generated code (.hpp) will be placed
            yk_codegen = namespace['yask-codegen'](name, 'devito',
                                                   configuration['platform'])
            if not os.path.exists(yk_codegen):
                os.makedirs(yk_codegen)

            # Write out the stencil file
            yk_codegen_file = os.path.join(yk_codegen, namespace['yask-codegen-file'])
            yc_soln.format(configuration['isa'], ofac.new_file_output(yk_codegen_file))

            # JIT-compile it
            compiler = configuration.yask['compiler']
            if configuration['develop-mode']:
                if yc_soln.get_num_equations() == 0:
                    # YASK will compile more quickly, and no price has to be paid
                    # in terms of performance, as this is a void kernel
                    opt_level = 0
                else:
                    opt_level = 1
            else:
                opt_level = 3
            args = [
                '-j', 'YK_CXX=%s' % compiler.cc, 'YK_CXXOPT=-O%d' % opt_level,
                # No MPI support at the moment
                'mpi=0',
                # To locate the YASK compiler
                'YC_EXEC=%s' % os.path.join(namespace['path'], 'bin'),
                # Error out if a grid not explicitly defined in the compiler is created
                'allow_new_grid_types=0',
                # To give a unique name to the generated Python modules, rather
                # than creating `yask_kernel.py`
                'YK_BASE=%s' % name,
                # `stencil` and `arch` should always be provided
                'stencil=%s' % 'devito', 'arch=%s' % configuration['platform'],
                # The root directory of generated code files, shared libs, Python modules
                'YASK_OUTPUT_DIR=%s' % namespace['yask-output-dir'],
                # Pick the YASK kernel Makefile, i.e. the one under `yask/src/kernel`
                '-C', namespace['kernel-path'], 'api'
            ]
            # Other potentially useful args:
            # - "EXTRA_MACROS=TRACE", -- debugging option
            make(namespace['path'], args)

            # Now we must be able to import the SWIG-generated Python module
            invalidate_caches()
            yk = import_module(name)

            # Release the lock manager
            cleanup_m.clean_up()

        # Create the YASK solution object
        kfac = yk.yk_factory()
        self.env = kfac.new_env()
        self.soln = kfac.new_solution(self.env)

        # Apply any user-provided options, if any.
        # These are applied here instead of just before prepare_solution()
        # so that applicable options will apply to all API calls.
        self.soln.apply_command_line_options(configuration.yask['options'] or '')

        # MPI setup: simple rank configuration in 1st dim only.
        # TODO: in production runs, the ranks would be distributed along all
        # domain dimensions.
        self.soln.set_num_ranks(self.space_dimensions[0], self.env.get_num_ranks())

        # Redirect stdout to a string or file
        if configuration.yask['dump']:
            filename = 'yk_dump.%s.%s.%s.txt' % (name, configuration['platform'],
                                                 configuration['isa'])
            filename = os.path.join(configuration.yask['dump'], filename)
            self.output = yk.yask_output_factory().new_file_output(filename)
        else:
            self.output = yk.yask_output_factory().new_string_output()
        self.soln.set_debug_output(self.output)

        # Users may want to run the same Operator (same domain etc.) with
        # different grids.
        self.grids = {i.get_name(): i for i in self.soln.get_grids()}
        self.local_grids = {i.name: self.grids[i.name] for i in (local_grids or [])}

    def new_grid(self, obj):
        """
        Create a new YASK grid.
        """
        return self.soln.new_fixed_size_grid('%s_%d' % (obj.name, contexts.ngrids),
                                             [str(i.root) for i in obj.indices],
                                             [int(i) for i in obj.shape])  # cast np.int

    def pre_apply(self, toshare):
        """
        Set up the YaskKernel before it's called from within an Operator.

        :param toshare: Mapper from functions to :class:`Data`s for sharing
                        grid storage.
        """
        # Sanity check
        grids = {i.grid for i in toshare if i.is_TensorFunction and i.grid is not None}
        assert len(grids) == 1
        grid = grids.pop()

        # Set the domain size, apply grid sharing, more sanity checks
        for k, v in zip(self.space_dimensions, grid.shape):
            self.soln.set_rank_domain_size(k, int(v))
        for k, v in toshare.items():
            target = self.grids.get(k.name)
            if target is not None:
                v._give_storage(target)
        assert all(not i.is_storage_allocated() for i in self.local_grids.values())
        assert all(v.is_storage_allocated() for k, v in self.grids.items()
                   if k not in self.local_grids)

        # Debug info
        debug("%s<%s,%s>" % (self.name, self.step_dimension, self.space_dimensions))
        for i in list(self.grids.values()) + list(self.local_grids.values()):
            if i.get_num_dims() == 0:
                debug("    Scalar: %s", i.get_name())
            elif not i.is_storage_allocated():
                size = [i.get_rank_domain_size(j) for j in self.space_dimensions]
                debug("    LocalGrid: %s%s, size=%s" %
                      (i.get_name(), str(i.get_dim_names()), size))
            else:
                size = []
                lpad, rpad = [], []
                for j in i.get_dim_names():
                    if j in self.space_dimensions:
                        size.append(i.get_rank_domain_size(j))
                        lpad.append(i.get_left_pad_size(j))
                        rpad.append(i.get_right_pad_size(j))
                    else:
                        size.append(i.get_alloc_size(j))
                        lpad.append(0)
                        rpad.append(0)
                debug("    Grid: %s%s, size=%s, left_pad=%s, right_pad=%s" %
                      (i.get_name(), str(i.get_dim_names()), size, lpad, rpad))

        # Set up the block shape for loop blocking
        for i, j in zip(self.space_dimensions, configuration.yask['blockshape']):
            self.soln.set_block_size(i, j)

        # This, amongst other things, allocates storage for the temporary grids
        self.soln.prepare_solution()

        # Set up auto-tuning
        if configuration['autotuning'].level is False:
            self.soln.reset_auto_tuner(False)
        elif configuration['autotuning'].mode == 'preemptive':
            self.soln.run_auto_tuner_now()

    def post_apply(self):
        """
        Release temporary storage and dump performance data about the last run.
        """
        # Release grid storage. Note: this *will not* cause deallocation, as these
        # grids are actually shared with the hook solution
        for i in self.grids.values():
            i.release_storage()
        # Release local grid storage. This *will* cause deallocation
        for i in self.local_grids.values():
            i.release_storage()
        # Dump performance data
        self.soln.get_stats()

    @property
    def space_dimensions(self):
        return tuple(self.soln.get_domain_dim_names())

    @property
    def step_dimension(self):
        return self.soln.get_step_dim_name()

    @property
    def rawpointer(self):
        return ctypes.cast(int(self.soln), namespace['type-solution'])

    def __repr__(self):
        return "YaskKernel [%s]" % self.name


class YaskContext(Signer):

    _hookcounter = 0
    """
    All of the shared objects generated by YASK for 'hook' solutions must
    have a unique name to avoid grid name clashes.
    """

    def __init__(self, name, grid):
        """
        Proxy between Devito and YASK.

        A ``YaskContext`` contains N :class:`YaskKernel` and M :class:`Data`,
        which have common space and time dimensions.

        :param name: Unique name of the context.
        :param grid: A :class:`Grid` carrying the context dimensions.
        """
        self.name = name
        self.space_dimensions = grid.dimensions
        self.step_dimension = grid.stepping_dim
        self.dtype = grid.dtype

        # All known solutions and grids in this context
        self.solutions = []
        self.grids = {}

    @property
    def dimensions(self):
        return (self.step_dimension,) + self.space_dimensions

    @property
    def ngrids(self):
        return len(self.grids)

    def make_grid(self, obj):
        """
        Create and return a new :class:`Data`, a YASK grid wrapper. Memory
        is allocated.

        :param obj: The :class:`Function` for which a YASK grid is allocated.
        """
        # 'hook' compiler solution: describes the grid
        # 'hook' kernel solution: allocates memory

        # A unique name for the 'hook' compiler and kernel solutions
        suffix = Signer._digest(self, obj, configuration, YaskContext._hookcounter)
        YaskContext._hookcounter += 1
        name = namespace['jit-hook'](suffix)

        # Create 'hook' compiler solution
        yc_hook = self.make_yc_solution(name)
        if obj.indices != self.dimensions:
            # Note: YASK wants *at least* a grid with *all* space (domain) dimensions
            # *and* the stepping dimension. `obj`, however, may actually employ a
            # different set of dimensions (e.g., a strict subset and/or some misc
            # dimensions). In such a case, an extra dummy grid is attached
            # `obj` examples: u(x, d), u(x, y, z)
            dimensions = [make_yask_ast(i, yc_hook, {}) for i in self.dimensions]
            yc_hook.new_grid('dummy_grid_full', dimensions)
        dimensions = [make_yask_ast(i.root, yc_hook, {}) for i in obj.indices]
        yc_hook.new_grid('dummy_grid_true', dimensions)

        # Create 'hook' kernel solution
        yk_hook = YaskKernel(name, yc_hook)
        grid = yk_hook.new_grid(obj)

        # Where should memory be allocated ?
        alloc = obj._allocator
        if alloc.is_Numa:
            if alloc.put_onnode:
                grid.set_numa_preferred(alloc.node)
            elif alloc.put_local:
                grid.set_numa_preferred(namespace['numa-put-local'])

        for i, s, h in zip(obj.indices, obj.shape_allocated, obj._extent_halo):
            if i.is_Space:
                # Note:
                # From the YASK docs: "If the halo is set to a value larger than
                # the padding size, the padding size will be automatically increased
                # to accomodate it."
                grid.set_left_halo_size(i.name, h.left)
                grid.set_right_halo_size(i.name, h.right)
            else:
                # time and misc dimensions
                assert grid.is_dim_used(i.root.name)
                assert grid.get_alloc_size(i.root.name) == s
        grid.alloc_storage()

        self.grids[grid.get_name()] = grid

        return grid

    def make_yc_solution(self, name):
        """
        Create and return a YASK compiler solution object.
        """
        yc_soln = cfac.new_solution(name)

        # Redirect stdout/strerr to a string or file
        if configuration.yask['dump']:
            filename = 'yc_dump.%s.%s.%s.txt' % (name, configuration['platform'],
                                                 configuration['isa'])
            filename = os.path.join(configuration.yask['dump'], filename)
            yc_soln.set_debug_output(ofac.new_file_output(filename))
        else:
            yc_soln.set_debug_output(ofac.new_null_output())

        # Set data type size
        yc_soln.set_element_bytes(self.dtype().itemsize)

        # Apply compile-time optimizations
        if configuration['isa'] != 'cpp':
            dimensions = [make_yask_ast(i, yc_soln, {}) for i in self.space_dimensions]
            # Vector folding
            for i, j in zip(dimensions, configuration.yask['folding']):
                yc_soln.set_fold_len(i, j)
            # Unrolling
            for i, j in zip(dimensions, configuration.yask['clustering']):
                yc_soln.set_cluster_mult(i, j)

        return yc_soln

    def make_yk_solution(self, name, yc_soln, local_grids):
        """
        Create and return a new :class:`YaskKernel` using ``yc_soln`` as
        compiler solution.
        """
        soln = YaskKernel(name, yc_soln, local_grids)
        self.solutions.append(soln)
        return soln

    def _signature_items(self):
        return tuple(i.name for i in self.dimensions)

    def __repr__(self):
        return ("YaskContext: %s\n"
                "- domain: %s\n"
                "- grids: [%s]\n"
                "- solns: [%s]\n") % (self.name, str(self.space_dimensions),
                                      ', '.join([i for i in list(self.grids)]),
                                      ', '.join([i.name for i in self.solutions]))


class ContextManager(OrderedDict):

    def __init__(self, *args, **kwargs):
        super(ContextManager, self).__init__(*args, **kwargs)
        self._partial_map = {}
        self._ncontexts = 0

    def _getkey(self, grid, dtype, dimensions=None):
        base = (configuration['isa'], dtype)
        if grid is not None:
            dims = filter_sorted((grid.time_dim, grid.stepping_dim) + grid.dimensions)
            return base + (tuple(dims),)
        elif dimensions:
            dims = filter_sorted([i for i in dimensions if i.is_Space])
            return base + (tuple(dims),)
        else:
            return base + ((),)

    def dump(self):
        """
        Drop all known contexts and clean up the relevant YASK directories.
        """
        self.clear()
        self._partial_map.clear()
        call(['rm', '-f'] + glob(os.path.join(namespace['path'], 'yask', '*hook*')))
        call(['rm', '-f'] + glob(os.path.join(namespace['path'], 'yask', '*soln*')))
        call(['rm', '-f'] + glob(os.path.join(namespace['path'], 'lib', '*hook*')))
        call(['rm', '-f'] + glob(os.path.join(namespace['path'], 'lib', '*soln*')))

    def fetch(self, dimensions, dtype):
        """
        Fetch the :class:`YaskContext` in ``self`` uniquely identified by
        ``dimensions`` and ``dtype``.
        """
        key = self._getkey(None, dtype, dimensions)

        context = self.get(key, self._partial_map.get(key))
        if context is not None:
            log("Fetched existing YaskContext from cache")
            return context
        else:
            exit("Couldn't find YaskContext for key=`%s`" % str(key))

    def putdefault(self, grid):
        """
        Derive a key ``K`` from the :class:`Grid` ``grid``; if ``K`` in ``self``,
        return the existing :class:`YaskContext` ``self[K]``, otherwise create a
        new context ``C``, set ``self[K] = C`` and return ``C``.
        """
        assert grid is not None

        key = self._getkey(grid, grid.dtype)

        # Does a YaskContext exist already corresponding to this key?
        if key in self:
            return self[key]

        # Functions declared with explicit dimensions (i.e., with no Grid) must be
        # able to retrieve the right context
        partial_keys = [self._getkey(None, grid.dtype, i) for i in powerset(key[-1])]
        if any(i in self._partial_map for i in partial_keys if i[2]):
            warning("Non-unique Dimensions found in different contexts; dumping "
                    "all known contexts. Perhaps you're attempting to use multiple "
                    "Grids, and some of them share identical Dimensions? ")
            self.dump()

        # Create a new YaskContext
        context = YaskContext('ctx%d' % self._ncontexts, grid)
        self._ncontexts += 1

        self[key] = context
        self._partial_map.update({i: context for i in partial_keys})

        log("Context successfully created!")

    @property
    def ngrids(self):
        return sum(i.ngrids for i in self.values())


contexts = ContextManager()
"""All known YASK contexts."""
