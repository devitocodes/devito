from ctypes import cdll
from function_manager import FunctionManager
from random import randint
from hashlib import sha1
import os
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
    _mic_flag = False
    # To enable mic process: 1:True 0:False;
    COMPILER_ENABLE_MIC = "DEVITO_MIC"
    _intel_compiler = ('icc', 'icpc')
    _device = None
    _stream = None
    _mic = None

    # The temp directory used to store generated code
    tmp_dir = os.path.join(gettempdir(), "devito-%s" % os.getuid())

    def __init__(self, propagators, dtype=None):
        self.compiler = guess_toolchain()
        override_var = os.environ.get(self.COMPILER_OVERRIDE_VAR, "")
        if override_var != "" and not override_var.isspace():
            self.compiler.cc = os.environ.get(self.COMPILER_OVERRIDE_VAR)
            enable_mic = os.environ.get(self.COMPILER_ENABLE_MIC)
            if self.compiler.cc in self._intel_compiler and enable_mic == "1":
                self._mic = __import__('pymic')
                self._mic_flag = True
        function_descriptors = [prop.get_fd() for prop in propagators]
        self.function_manager = FunctionManager(function_descriptors, self._mic_flag)
        self._function_descriptors = function_descriptors
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
        if self._mic_flag:
                self.compiler.ldflags.append("-mmic")
        jit.extension_file_from_string(self.compiler, self.src_lib,
                                       self.src_code, source_name=self.src_file)

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
