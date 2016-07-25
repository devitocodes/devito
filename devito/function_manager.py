import numpy as np

import cgen_wrapper as cgen
from devito.tools import convert_dtype_to_ctype


class FunctionManager(object):
    """Class that accepts a list of :class:`FunctionDescriptor` objects and generates the C
    function represented by it

    :param function_descriptor: The list of :class:`FunctionDescriptor` objects
    :param mic_flag: True if using MIC. Default is False
    :param openmp: True if using OpenMP. Default is False
    """
    libraries = ['cassert', 'cstdlib', 'cmath', 'iostream',
                 'fstream', 'vector', 'cstdio', 'string', 'inttypes.h', 'sys/time.h', 'math.h']

    _pymic_attribute = 'PYMIC_KERNEL'

    def __init__(self, function_descriptors, mic_flag=False, openmp=False):
        self.function_descriptors = function_descriptors
        self._defines = []
        self.mic_flag = mic_flag

        if openmp:
            self.libraries = self.libraries + ['omp.h']

    def includes(self):
        """
        :returns: :class:`cgen.Module` -- A module containing all the preprocessor directives
        """
        statements = []
        statements += self._defines
        statements += [cgen.Include(s) for s in self.libraries]

        if self.mic_flag:
            statements += [cgen.Include('pymic_kernel.h')]

        return cgen.Module(statements)

    def add_define(self, name, text):
        """Adds a #define directive to the preprocessor

        :param name: The symbol to be replaced
        :param text: The value that replaces it
        """
        self._defines.append(cgen.Define(name, text))

    def generate(self):
        """:returns: :class:`cgen.Module` -- A module containing the includes and the generated functions
        """
        statements = [self.includes()]
        statements += [self.process_function(m) for m in self.function_descriptors]

        return cgen.Module(statements)

    def process_function(self, function_descriptor):
        """Generates a function signature and body from a :class:`FunctionDescriptor`

        :param function_descriptor: The :class:`FunctionDescriptor` to process
        :returns: :class:`cgen.FunctionBody` -- The function body generated from the function_descriptor
        """
        return cgen.FunctionBody(self.generate_function_signature(function_descriptor),
                                 self.generate_function_body(function_descriptor))

    def generate_function_signature(self, function_descriptor):
        """Generates a function signature from a :class:`FunctionDescriptor`

        :param function_descriptor: The :class:`FunctionDescriptor` to process
        :returns: :class:`cgen.FunctionDeclaration` -- The function declaration generated from function_descriptor
        """
        function_params = []

        for param in function_descriptor.matrix_params:
            param_vec_def = cgen.Pointer(cgen.POD(param['dtype'], param['name']+"_vec"))
            function_params = function_params + [param_vec_def]

        if self.mic_flag:
            function_params += [cgen.Pointer(cgen.POD(type_label, name+"_pointer"))
                                for type_label, name in function_descriptor.value_params]

            return cgen.FunctionDeclaration(cgen.Value(self._pymic_attribute + '\nint', function_descriptor.name),
                                            function_params)
        else:
            function_params += [cgen.POD(type_label, name) for type_label, name in function_descriptor.value_params]

            return cgen.Extern("C",
                               cgen.FunctionDeclaration(cgen.Value('int', function_descriptor.name), function_params))

    def generate_function_body(self, function_descriptor):
        """Generates a function body from a :class:`FunctionDescriptor`

        :param function_descriptor: The :class:`FunctionDescriptor` to process
        :returns: :class:`cgen.Block` -- A block containing the generated function body
        """
        statements = [cgen.POD(var[0], var[1]) for var in function_descriptor.local_vars]

        for param in function_descriptor.matrix_params:
            num_dim = len(param['shape'])
            arr = "".join(
                ["[%d]" % (param['shape'][i])
                 for i in range(1, num_dim)]
            )
            cast_pointer = cgen.Initializer(
                cgen.POD(param['dtype'], "(*%s)%s" % (param['name'], arr)),
                '(%s (*)%s) %s' % (cgen.dtype_to_ctype(param['dtype']), arr, param['name']+"_vec")
            )
            statements.append(cast_pointer)

        if self.mic_flag:
            for param in function_descriptor.value_params:
                cast_pointer = cgen.Initializer(cgen.POD(param[0], "(%s)" % (param[1])), '*%s' % (param[1]+"_pointer"))
                statements.append(cast_pointer)

        statements.append(function_descriptor.body)
        statements.append(cgen.Statement("return 0"))

        return cgen.Block(statements)


class FunctionDescriptor(object):
    """This class can be used to describe a general C function which receives
    multiple parameters and loops over one of the parameters, executing a
    given body of statements inside the loop

    :param name: The name of the function
    """
    def __init__(self, name):
        self.name = name
        self.matrix_params = []
        self.value_params = []
        self.local_vars = []

    def add_matrix_param(self, name, shape, dtype):
        """Add a matrix parameter to the function

        :param name: The name of the matrix
        :param shape: Tuple of matrix dimensions
        :param dtype: The :class:`numpy.dtype` of the matrix
        """
        self.matrix_params.append({'name': name, 'shape': shape, 'dtype': dtype})

    def add_value_param(self, name, dtype):
        """Declare a new value (scalar) parameter for this function

        :param name: The name of the scalar
        :param dtype: The :class:`numpy.dtype` of the scalar
        """
        self.value_params.append((np.dtype(dtype), name))

    def add_local_variable(self, name, dtype):
        """Add a local variable to the function

        :param name: The name of the local variable
        :param dtype: The :class:`numpy.dtype` of the variable
        """
        try:
            self.local_vars.append((np.dtype(dtype), name))
        except:
            self.local_vars.append((dtype, name))

    def set_body(self, body):
        """Sets the body of the function

        :param body: A :class:`cgen.Block` containing the body of the function
        """
        self.body = body

    @property
    def params(self):
        """:returns: A list of all the matrix and scalar parameters
        """
        return self.matrix_params + self.value_params

    @property
    def argtypes(self):
        """Create argument types for defining function signatures via ctypes

        :returns: A list of ctypes of the matrix parameters and scalar parameters
        """
        argtypes = [np.ctypeslib.ndpointer(dtype=p['dtype'], flags='C')
                    for p in self.matrix_params]
        argtypes += [convert_dtype_to_ctype(p[0]) for p in self.value_params]

        return argtypes
