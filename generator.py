from ctypes import cdll, c_int
import numpy as np
from function_manager import FunctionManager
from random import randint
from hashlib import sha1
import os
from _ctypes import ArgumentError
import cgen_wrapper as cgen
import function_descriptor
from codepy.toolchain import guess_toolchain
import codepy.jit as jit
from tempfile import gettempdir


class Generator(object):
    """ This is the primary interface class for code
    generation. However, the code in this class is focused on
    interfacing with the generated code. The actual code generation
    happens in BasicTemplate
    """
    _hashing_function = sha1
    _wrapped_functions = None
    # The temp directory used to store generated code
    tmp_dir = os.path.join(gettempdir(), "devito-%s" % os.getuid())

    def __init__(self, function_descriptors, dtype = None):
        self.function_manager = FunctionManager(function_descriptors)
        self._function_descriptors = function_descriptors
        self.compiler = guess_toolchain()
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
    def function_descriptor(self):
        return self._function_descriptor

    # If the function descriptor is changed, invalidate the cache and regenerate and recompile the code
    @function_descriptor.setter
    def function_descriptors(self, function_descriptors):
        self._function_descriptors = function_descriptors
        self.__clean()

    # Add a C macro to the generated code
    def add_macro(self, name, text):
        self.cgen_template.add_define(name, text)

    def compile(self):
        # Generate compilable source code
        self.src_code = str(self.function_manager.generate())
        print "Generated: %s" % self.src_file
        jit.extension_file_from_string(self.compiler, self.src_lib,
                                       self.src_code, source_name=self.src_file)

    """ Wrap the function by converting the python style arguments(simply passing object references)
        to C style (pointer + int dimensions)
    """
    def wrap_function(self, function, function_descriptor):
        def wrapped_function(*args):
            num_params = len(function_descriptor.params)
            assert len(args) == num_params, "Expected %d parameters, got %d" % (num_params, len(args))
            arg_list = []
            for i, param in zip(range(num_params), function_descriptor.params):
                try:
                    param_shape = args[i].shape  # Assume that param is a matrix param - fail otherwise
                    param_ref = np.asarray(args[i], dtype=self.dtype)  # Force the parameter provided as argument to be ndarray (not tuple or list)
                except:  # Param is a scalar
                    param_ref = args[i]  # No conversion necessary for a scalar value
                    param_shape = []
                arg_list += [param_ref] + list(param_shape)
            try:
                function(*arg_list)
            except ArgumentError as inst:
                assert False, "Argument Error (%s), provided arguments: " % str(inst)+" ,".join([str(type(arg)) for arg in arg_list])

        return wrapped_function

    def _prepare_wrapped_function(self, function_descriptor, compiler='g++'):
        # Compile code if this hasn't been done yet
        self.compile()
        # Load compiled binary
        self.__load_library(src_lib=self.src_lib)

        # Type: double* in C
        array_nd_double = np.ctypeslib.ndpointer(dtype=self.dtype, flags='C')

        # Pointer to the function in the compiled library
        library_function = getattr(self._library, function_descriptor.name)

        # Ctypes needs an array describing the function parameters, prepare that array
        argtypes = []
        for param in function_descriptor.params:

            try:
                num_dim = param.get('num_dim')  # Assume that param is a matrix param - fail otherwise
                # Pointer to the parameter
                argtypes.append(array_nd_double)
                # Ints for the sizes of the parameter in each dimension
                argtypes += [c_int for i in range(0, num_dim)]
            except:
                # Param is a value param
                argtypes.append(cgen.convert_dtype_to_ctype(param[0])) # There has to be a better way of doing this
        library_function.argtypes = argtypes

        return self.wrap_function(library_function, function_descriptor)

    def get_wrapped_functions(self):
        if self._wrapped_functions is None:
            self._wrapped_functions = [self._prepare_wrapped_function(fd) for fd in self._function_descriptors]
        return self._wrapped_functions

    def __clean(self):
        self._wrapped_functions = None
        self.src_lib = None
        self.src_file = None
        self.src_lib = None
        self.filename = None
