from compilation import GNUCompiler, ClangCompiler, IntelCompiler
from ctypes import cdll, c_int
import numpy as np
from templates.basic_template import BasicTemplate
from random import randint
from hashlib import sha1
import os

# This is the primary interface class for code generation. However, the code in this class is focused on interfacing with the
# generated code. The actual code generation happens in BasicTemplate
class Generator(object):
    src_lib = None
    src_file = None
    _hash = sha1
    _wrapped_function = None
    # The temp directory used to store generated code
    _tmp_dir_name = "tmp"
    def __init__(self, function_descriptor):
        self.cgen_template = BasicTemplate(function_descriptor)
        self._function_descriptor = function_descriptor
        self._compiler = GNUCompiler()
        # Generate a random salt to uniquely identify this instance of the class
        self._salt = randint(0, 100000000)
        self.__generate_filename()
        # If the temp does not exist, create it
        if not os.path.isdir(self._tmp_dir_name):
            os.mkdir(self._tmp_dir_name)
    
    def __generate_filename(self):
        # Generate a unique filename for the generated code by combining the unique salt 
        # with the hash of the parameters for the function as well as the body of the function
        filename = self._tmp_dir_name+"/"+self._hash(str(self._salt)+str(self._function_descriptor.params)+str(self._function_descriptor.body)).hexdigest()+".cpp"
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
    def function_descriptor(self):
        return self._function_descriptor
    
    # If the function descriptor is changed, invalidate the cache and regenerate and recompile the code
    @function_descriptor.setter
    def function_descriptor(self, function_descriptor):
        self._function_descriptor = function_descriptor
        self.__clean()
    
    # Add a C macro to the generated code
    def add_macro(self, name, text):
        self.cgen_template.add_define(name, text)
    
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
    
    # Wrap the function by converting the python style arguments(simply passing object references)
    # to C style (pointer + int dimensions)
    def wrap_function(self, function):
        def wrapped_function(*args):
            num_params = len(self._function_descriptor.params)
            assert(len(args)==num_params)
            arg_list = []
            for i, param in zip(range(num_params), self._function_descriptor.params):
                arg_list+= [args[i]] + list(param['shape'])
            function(*arg_list)
        return wrapped_function

    def _prepare_wrapped_function(self, compiler='g++'):
        # Compile code if this hasn't been done yet
        if self.src_lib is None:
            self.compile(compiler=compiler, shared=True)
        # Load compiled binary
        self.__load_library(src_lib=self.src_lib)
        
        # Type: double* in C
        array_nd_double = np.ctypeslib.ndpointer(dtype=np.double, flags='C')
        
        # Pointer to the function in the compiled library
        library_function = getattr(self._library, self._function_descriptor.name)
        
        # Ctypes needs an array describing the function parameters, prepare that array
        argtypes = []
        for param in self._function_descriptor.params:
            # Pointer to the parameter
            argtypes.append(array_nd_double)
            # Ints for the sizes of the parameter in each dimension
            argtypes += [c_int for i in range(0, len(param['shape']))]
        library_function.argtypes = argtypes
        
        self._wrapped_function = self.wrap_function(library_function)
        
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
        self.filename = None
        
