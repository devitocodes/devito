from numpy.ctypeslib import ctypes
from compilation import *
from ctypes import cdll, Structure, POINTER, c_float, pointer, c_longlong, c_int
import numpy as np
from templates.basic_template import BasicTemplate

class Generator(object):
    src_lib = None
    src_file = None
    def __init__(self, arg_grid):
        self.cgen_template = BasicTemplate(len(arg_grid.shape))
        self._compiler = GNUCompiler()
        self._arg_grid = arg_grid
          
    def _load_library(self, src_lib):
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
    
    def generate(self, filename, compiler=None):
        if compiler:
            self.compiler = compiler

        self.src_code = str(self.cgen_template.generate())
        # Generate compilable source code
        self.src_file = filename
        with file(self.src_file, 'w') as f:
            f.write(self.src_code)

        print "Generated:", self.src_file
        
    def compile(self, filename, compiler=None, shared=True):
        if compiler:
            self.compiler = compiler

        # Generate code if this hasn't been done yet
        if self.src_file is None:
            self.generate(filename)

        # Compile source file
        out = self.compiler.compile(self.src_file, shared=shared)
        if shared:
            self.src_lib = out
        return out
    
    def create_ptr_array(self, x):
        # The last line of this method was inspired by http://numpy-discussion.10968.n7.nabble.com/Pass-2d-ndarray-into-C-double-using-ctypes-td39414.html
        # This method is an attempt to generalise that process to n dimensional arrays
        dim = len(x.shape)-2
        length = 1
        for i in range(dim, -1, -1):
            length = length*x.shape[i]
        print "dim: "+str(dim)+", length: "+str(length)
        return x.__array_interface__['data'][dim] + np.arange(length)*x.strides[dim]
    
    def wrap_function(self, function):
        def wrapped_function(x):
            y = np.empty_like(x)
            shape = x.shape
            print x.strides
            xpp = self.create_ptr_array(x).astype(np.uintp) 
            ypp = self.create_ptr_array(y).astype(np.uintp) 
            
            arg_list = [xpp, ypp] + list(shape)
            print arg_list
            function(*arg_list)
            return y
        return wrapped_function
    
    def execute(self, filename, compiler='clang'):

        # Compile code if this hasn't been done yet
        if self.src_lib is None:
            self.compile(filename, compiler=compiler, shared=True)
        # Load compiled binary
        self._load_library(src_lib=self.src_lib)
        array_1d_double = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
        opesci_process = self._library.opesci_process
        opesci_process.argtypes = [array_1d_double, array_1d_double] + [c_int for i in range(1, len(self._arg_grid.shape))]
        
        wrapped_function = self.wrap_function(opesci_process)
        
        
        ret = wrapped_function(self._arg_grid)
        print ret
        