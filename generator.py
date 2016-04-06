from compilation import GNUCompiler, ClangCompiler, IntelCompiler
from ctypes import cdll, c_int
import numpy as np
from templates.basic_template import BasicTemplate
from random import randint
from hashlib import sha1
import os

class Generator(object):
    src_lib = None
    src_file = None
    _hash = sha1
    _wrapped_function = None
    _tmp_dir_name = "tmp"
    def __init__(self, arg_shape, kernel, skip_elements = None):
        self.cgen_template = BasicTemplate(len(arg_shape), kernel, skip_elements)
        self._kernel = kernel
        self._compiler = GNUCompiler()
        self._arg_shape = arg_shape
        self._salt = randint(0, 100000000)
        self.__generate_filename()
        if not os.path.isdir(self._tmp_dir_name):
            os.mkdir(self._tmp_dir_name)
    
    def __generate_filename(self):
        filename = self._tmp_dir_name+"/"+self._hash(str(self._salt)+str(self._arg_shape)+str(self._kernel)).hexdigest()+".cpp"
        self._filename = filename
    
    def __load_library(self, src_lib):
        """Load a compiled dynamic binary using ctypes.cdll"""
        libname = src_lib or self.src_lib
        try:
            self._library = cdll.LoadLibrary(libname)
        except OSError as e:
            print "Library load error: ", e
            raise Exception("Failed to load %s" % libname)

    @property
    def compiler(self):
        return self._compiler

    @compiler.setter
    def compiler(self, compiler):
        if compiler in ['g++', 'gnu']:
            self._compiler = GNUCompiler()
        elif compiler in ['icpc', 'intel']:
            self._compiler = IntelCompiler()
        elif compiler in ['clang', 'clang++']:
            self._compiler = ClangCompiler()
        else:
            raise ValueError("Unknown compiler.")
    
    @property
    def kernel(self):
        return self._kernel
    
    @kernel.setter
    def kernel(self, kernel):
        self._kernel = kernel
        self.__clean()
    
    @property
    def arg_shape(self):
        return self._arg_shape
    
    @arg_shape.setter
    def arg_shape(self, arg_shape):
        self._arg_shape = arg_shape
        self.__clean()
    
    def add_macro(self, function_name, function_text):
        self.cgen_template.add_define(function_name, function_text)
    
    def generate(self, compiler=None):
        if compiler:
            self.compiler = compiler

        self.src_code = str(self.cgen_template.generate())
        # Generate compilable source code
        self.src_file = self._filename
        with file(self.src_file, 'w') as f:
            f.write(self.src_code)

        print "Generated:", self.src_file

    def compile(self, compiler=None, shared=True):
        if compiler:
            self.compiler = compiler

        # Generate code if this hasn't been done yet
        if self.src_file is None:
            self.generate()

        # Compile source file
        out = self.compiler.compile(self.src_file, shared=shared)
        if shared:
            self.src_lib = out
        return out

    def wrap_function(self, function):
        def wrapped_function(x):
            y = np.empty_like(x)
            arg_list = [x, y] + list(x.shape)
            function(*arg_list)
            return y
        return wrapped_function

    def _prepare_wrapped_function(self, compiler='g++'):
        # Compile code if this hasn't been done yet
        if self.src_lib is None:
            self.compile(compiler=compiler, shared=True)
        # Load compiled binary
        self.__load_library(src_lib=self.src_lib)
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, flags='C')
        opesci_process = self._library.opesci_process
        opesci_process.argtypes = [array_1d_double, array_1d_double] + [c_int for i in range(1, len(self._arg_shape))]
        self._wrapped_function = self.wrap_function(opesci_process)
        
    def get_wrapped_function(self):
        if self._wrapped_function is None:
            if self._filename is None:
                self.__generate_filename()
            self._prepare_wrapped_function()
        return self._wrapped_function
    
    def __clean(self):
        self._wrapped_function = None
        self.src_lib = None
        self.src_file = None
        self.src_lib = None
        
