from ctypes import cdll
import numpy as np
from function_manager import FunctionManager
from random import randint
from hashlib import sha1
import os
from _ctypes import ArgumentError
import cgen_wrapper as cgen
from codepy.toolchain import guess_toolchain
import codepy.jit as jit
from tempfile import gettempdir


class JitManager(object):
    """ This is the primary interface class for code
    generation. However, the code in this class is focused on
    interfacing with the generated code. The actual code generation
    happens in FunctionManager
    """
    _hashing_function = sha1
    _wrapped_functions = None
    COMPILER_OVERRIDE_VAR = "DEVITO_CC"
    _incompatible_flags = ["-Wshorten-64-to-32", "-Wstrict-prototypes", ("-arch", "i386")]
    
    # The temp directory used to store generated code
    tmp_dir = os.path.join(gettempdir(), "devito-%s" % os.getuid())

    def __init__(self, propagators, dtype=None):
        function_descriptors = [prop.get_fd() for prop in propagators]
        self.function_manager = FunctionManager(function_descriptors)
        self._function_descriptors = function_descriptors
        self.compiler = guess_toolchain()
        if os.environ.get(self.COMPILER_OVERRIDE_VAR, "") != "":
            self.compiler.cc = os.environ.get(self.COMPILER_OVERRIDE_VAR)
        self._clean_flags()
        # Generate a random salt to uniquely identify this instance of the class
        self._salt = randint(0, 100000000)
        self._basename = self.__generate_filename()
        self.src_file = os.path.join(self.tmp_dir, "%s.cpp" % self._basename)
        self.src_lib = os.path.join(self.tmp_dir, "%s.so" % self._basename)
        self.dtype = dtype
        # If the temp does not exist, create it
        if not os.path.isdir(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def __generate_filename(self):
        # Generate a unique filename for the generated code by combining the unique salt
        # with the hash of the parameters for the function as well as the body of the function
        hash_string = "".join([str(fd.params) for fd in self._function_descriptors])
        self._hash = self._hashing_function(hash_string).hexdigest()
        return self._hash

    def __load_library(self, src_lib):
        """Load a compiled dynamic binary using ctypes.cdll"""
        libname = src_lib or self.src_lib
        try:
            self._library = cdll.LoadLibrary(libname)
        except OSError as e:
            print "Library load error: ", e
            raise Exception("Failed to load %s" % libname)

    @property
    def function_descriptors(self):
        return self._function_descriptors

    # If the function descriptor is changed, invalidate the cache and regenerate and recompile the code
    @function_descriptors.setter
    def function_descriptors(self, function_descriptors):
        self._function_descriptors = function_descriptors
        self.__clean()

    def compile(self):
        # Generate compilable source code
        self.src_code = str(self.function_manager.generate())
        print "Generated: %s" % self.src_file
        jit.extension_file_from_string(self.compiler, self.src_lib,
                                       self.src_code, source_name=self.src_file)

    def wrap_function(self, function, function_descriptor):
        """ Wrap the function by converting the python style arguments(simply passing object references)
            to C style (pointer + int dimensions)
        """
        def wrapped_function(*args):
            num_params = len(function_descriptor.params)
            assert len(args) == num_params, "Expected %d parameters, got %d" % (num_params, len(args))
            arg_list = []
            num_matrix_params = len(function_descriptor.matrix_params)
            for i in range(num_params):
                if i < num_matrix_params:
                    param_ref = np.asarray(args[i], dtype=self.dtype)  # Force the parameter provided as argument to be ndarray (not tuple or list)
                else:
                    param_ref = args[i]  # No conversion necessary for a scalar value
                arg_list += [param_ref]
            try:
                function(*arg_list)
            except ArgumentError as inst:
                assert False, "Argument Error (%s), provided arguments: " % str(inst)+" ,".join([str(type(arg)) for arg in arg_list])

        return wrapped_function

    def _prepare_wrapped_functions(self):
        # Compile code if this hasn't been done yet
        self.compile()
        # Load compiled binary
        self.__load_library(src_lib=self.src_lib)

        # Type: double* in C
        array_nd = np.ctypeslib.ndpointer(dtype=self.dtype, flags='C')
        wrapped_functions = []
        for function_descriptor in self._function_descriptors:
            # Pointer to the function in the compiled library
            library_function = getattr(self._library, function_descriptor.name)
            # Ctypes needs an array describing the function parameters, prepare that array
            argtypes = [array_nd for i in function_descriptor.matrix_params]
            argtypes += [cgen.convert_dtype_to_ctype(param[0]) for param in function_descriptor.value_params]
            library_function.argtypes = argtypes
            wrapped_functions.append(self.wrap_function(library_function, function_descriptor))
        return wrapped_functions

    def get_wrapped_functions(self):
        if self._wrapped_functions is None:
            self._wrapped_functions = self._prepare_wrapped_functions()
        return self._wrapped_functions

    def __clean(self):
        self._wrapped_functions = None
        self.src_lib = None
        self.src_file = None
        self.src_lib = None
        self.filename = None

    def _clean_flags(self):
        for flag in self._incompatible_flags:
            for flag_list in [self.compiler.cflags, self.compiler.ldflags]:
                if isinstance(flag, tuple):
                    to_delete = []
                    for i, item in enumerate(flag_list):
                        if flag_list[i].strip() == flag[0] and flag_list[i+1].strip() == flag[1]:
                            to_delete.append(i)
                            to_delete.append(i+1)
                    for i in sorted(to_delete, reverse=True):
                        del flag_list[i]
                else:
                    while flag in flag_list:
                        flag_list.remove(flag)
