from ctypes import cdll
from function_manager import FunctionManager, FunctionDescriptor
from random import randint
from hashlib import sha1
import os
from codepy.toolchain import guess_toolchain
import codepy.jit as jit
from tempfile import gettempdir
import numpy as np
from tools import convert_dtype_to_ctype
import cgen_wrapper as cgen


class JitManager(object):
    _hashing_function = sha1
    _wrapped_functions = None
    COMPILER_OVERRIDE_VAR = "DEVITO_CC"
    REMOVE_FLAG_VAR = "DEVITO_REMOVE_FLAG"
    _incompatible_flags = ["-O2", "-Wshorten-64-to-32", "-Wstrict-prototypes", ("-arch", "i386")]
    _compatible_flags = ["-O3", "-g"]
    _mic_flag = False
    # To enable mic process: 1:True 0:False;
    COMPILER_ENABLE_MIC = "DEVITO_MIC"
    _intel_compiler = ('icc', 'icpc')
    _device = None
    _stream = None
    _mic = None

    # The temp directory used to store generated code
    tmp_dir = os.path.join(gettempdir(), "devito-%s" % os.getuid())

    def __init__(self, propagators, dtype=None, openmp=False):
        self.compiler = guess_toolchain()
        self._openmp = openmp
        override_var = os.environ.get(self.COMPILER_OVERRIDE_VAR, "")
        if override_var != "" and not override_var.isspace():
            self.compiler.cc = override_var
            enable_mic = os.environ.get(self.COMPILER_ENABLE_MIC)
            if self.compiler.cc in self._intel_compiler and enable_mic == "1":
                self._mic = __import__('pymic')
                self._mic_flag = True

        remove_flags = os.environ.get(self.REMOVE_FLAG_VAR, "")
        if remove_flags != "" and not remove_flags.isspace():
            flags = remove_flags.split(":")
            self._incompatible_flags = self._incompatible_flags + flags

        function_descriptors = [prop.get_fd() for prop in propagators]

        # add main method if auto tuning to produce executable
        if propagators[0].auto_tune and propagators[0].cache_blocking:
            self._append_main_function(function_descriptors, propagators)

        self.function_manager = FunctionManager(function_descriptors, self._mic_flag, self._openmp)
        self._function_descriptors = function_descriptors
        self._clean_flags()
        self._add_flags()
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
        hash_string = "".join([str(fd.params) for fd in self._function_descriptors]) + str(self._salt)
        self._hash = self._hashing_function(hash_string).hexdigest()
        return self._hash

    def __load_library(self, src_lib):
        """Load a compiled dynamic binary using ctypes.cdll"""
        libname = src_lib or self.src_lib
        try:
            if self._mic_flag:
                # Load pymic objects to perform offload process
                self._device = self._mic.devices[0]
                self._stream = self._device.get_default_stream()
                self.library = self._device.load_library(libname)
            else:
                self.library = cdll.LoadLibrary(libname)
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

    def __clean(self):
        self._wrapped_functions = None
        self.src_lib = None
        self.src_file = None
        self.src_lib = None
        self.filename = None

    # adds main function to function desc for auto tuning.
    def _append_main_function(self, function_descriptors, propagators):
        statements = []  # statements for cgen.block
        pnames = []
        main_fd = FunctionDescriptor("main")
        main_fd.return_type = "int"

        # allocates the space for matrix'es
        # Note currently auto tunes only the first function in function descriptors. If scope is larger. Extend
        for param in function_descriptors[0].matrix_params:
            array_size_str = ""
            for shape in param["shape"]:
                array_size_str += "%s * " % shape

            ptype = cgen.dtype_to_ctype(param['dtype'])
            pname = param["name"] + "_vec"
            pnames.append(pname)

            # Produces similar str: double* m_vec =(double*)malloc(336*336*336*sizeof(double))
            all_str = "%s* %s = (%s*)malloc(%ssizeof(%s))" % (ptype, pname, ptype, array_size_str, ptype)
            statements.append(cgen.Statement(all_str))

        statements.append(cgen.Pragma("isat marker %s" % propagators[0].at_markers[1][0]))  # tells at measure start

        #                      cuts the [    removes ]        removes ' symbol
        function_args_str = str(pnames)[1:].replace(']', '').replace('\'', '')
        statements.append(cgen.Statement("%s(%s)" % (function_descriptors[0].name, function_args_str)))

        statements.append(cgen.Pragma("isat marker %s" % propagators[0].at_markers[1][1]))  # tells at measure end

        main_fd.set_body(cgen.Block(statements))
        function_descriptors.append(main_fd)

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

    def _add_flags(self):
        if self.compiler.cc in self._intel_compiler:
            if self._openmp:
                self.compiler.ldflags.append("-openmp")
            if self._mic_flag:
                self.compiler.ldflags.append("-mmic")
            else:
                self.compiler.ldflags.append("-mavx")
        else:
            if self._openmp:
                self.compiler.cflags.append("-fopenmp")
        for flag in self._compatible_flags:
            self.compiler.ldflags.append(flag)
            self.compiler.cflags.append(flag)

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

    def prepare_wrapped_function(self):
        # Compile code if this hasn't been done yet
        self.compile()
        # Load compiled binary
        self.__load_library(src_lib=self.src_lib)

        # Type: double* in C
        array_nd = np.ctypeslib.ndpointer(dtype=self.dtype, flags='C')
        wrapped_functions = []
        for function_descriptor in self._function_descriptors:
            # Pointer to the function in the compiled library
            library_function = getattr(self.library, function_descriptor.name)
            if self._mic_flag:
                wrapped_functions.append(self.create_a_function(self._stream, library_function))
            else:
                # Ctypes needs an array describing the function parameters, prepare that array
                argtypes = [array_nd for i in function_descriptor.matrix_params]
                argtypes += [convert_dtype_to_ctype(param[0]) for param in function_descriptor.value_params]
                library_function.argtypes = argtypes
                wrapped_functions.append(self.wrap_function(library_function, function_descriptor))
        return wrapped_functions

    def create_a_function(self, mic_stream, mic_library):
        # Create function dynamically to execute mic offload using pymic
        def function_template(*args, **kwargs):
            mic_stream.invoke(mic_library, args)
            mic_stream.sync()
        return function_template

    def get_wrapped_functions(self):
        if self._wrapped_functions is None:
            self._wrapped_functions = self.prepare_wrapped_function()
        return self._wrapped_functions
