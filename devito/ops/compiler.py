from codepy.jit import compile_from_string
from time import time
import warnings

from devito.compiler import Compiler
from devito.logger import debug, warning, error
from devito.parameters import configuration
from devito.tools import (as_tuple, change_directory, filter_ordered,
                          memoized_meth, make_tempdir)
__all__ = ['CompilerOPS']

class CompilerOPS(configuration['compiler'].__class__):
    def __init__(self, *args, **kwargs):
        super(CompilerOPS, self).__init__(*args, **kwargs)

    def jit_compile_ops(self, soname, code, hcode):
        target = str(self.get_jit_dir().joinpath(soname))
        src_file = "%s.%s" % (target, self.src_ext)
        src_file2 = "%s.h" % (target)
        cache_dir = self.get_codepy_dir().joinpath(soname[:7])
        if configuration['jit-backdoor'] is False:
            # Typically we end up here
            # Make a suite of cache directories based on the soname
            cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Warning: dropping `code` on the floor in favor to whatever is written
            # within `src_file`
            try:
                with open(src_file, 'r') as f:
                    code = f.read()
                # Bypass the devito JIT cache
                # Note: can't simply use Python's `mkdtemp()` as, with MPI, different
                # ranks would end up creating different cache dirs
                cache_dir = cache_dir.joinpath('jit-backdoor')
                cache_dir.mkdir(parents=True, exist_ok=True)
            except FileNotFoundError:
                raise ValueError("Trying to use the JIT backdoor for `%s`, but "
                                "the file isn't present" % src_file)

        # `catch_warnings` suppresses codepy complaining that it's taking
        # too long to acquire the cache lock. This warning can only appear
        # in a multiprocess session, typically (but not necessarily) when
        # many processes are frequently attempting jit-compilation (e.g.,
        # when running the test suite in parallel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            tic = time()
            # Spinlock in case of MPI
            sleep_delay = 0 if configuration['mpi'] else 1
            _, _, _, recompiled = compile_from_string(self, target, code, src_file,
                                                    cache_dir=cache_dir,
                                                    debug=configuration['debug-compiler'],
                                                    sleep_delay=sleep_delay)
            _, _, _, recompiled_kernel = compile_from_string(self, target, hcode, src_file2,
                                                    cache_dir=cache_dir,
                                                    debug=configuration['debug-compiler'],
                                                    sleep_delay=sleep_delay)

            toc = time()

        if recompiled and recompiled_kernel:
            debug("%s: compiled `%s` [%.2f s]" % (self, src_file, toc-tic))
        elif not recompiled and recompiled_kernel:
            debug("%s: cache hit `%s` [%.2f s]" % (self, src_file, toc-tic))
        elif recompiled and not recompiled_kernel:
            debug("%s: cache hit `%s` [%.2f s]" % (self, src_file2, toc-tic))
        else:
            debug("%s: cache hit `%s` and `%s` [%.2f s]" % (self, src_file, src_file2, toc-tic))
