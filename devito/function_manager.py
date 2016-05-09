import cgen_wrapper as cgen
import numpy as np


class FunctionManager(object):
    """Class that accepts a list of FunctionDescriptor objects and generates the C
        function represented by it
    """
    libraries = ['cassert', 'cstdlib', 'cmath', 'iostream',
                 'fstream', 'vector', 'cstdio', 'string', 'inttypes.h']

    _pymic_attribute = 'PYMIC_KERNEL'

    def __init__(self, function_descriptors, mic_flag=False):
        self.function_descriptors = function_descriptors
        self._defines = []
        self.mic_flag = mic_flag

    def includes(self):
        statements = []
        statements += self._defines
        statements += [cgen.Include(s) for s in self.libraries]
        if self.mic_flag:
            statements += [cgen.Include('pymic_kernel.h')]
        return cgen.Module(statements)

    def add_define(self, name, text):
        self._defines.append(cgen.Define(name, text))

    def generate(self):
        statements = [self.includes()]
        statements += [self.process_function(m) for m in self.function_descriptors]
        return cgen.Module(statements)

    def process_function(self, function_descriptor):
        return cgen.FunctionBody(self.generate_function_signature(function_descriptor),
                                 self.generate_function_body(function_descriptor))

    def generate_function_signature(self, function_descriptor):
        function_params = []
        for param in function_descriptor.matrix_params:
            param_vec_def = cgen.Pointer(cgen.POD(param['dtype'], param['name']+"_vec"))
            function_params = function_params + [param_vec_def]
        if self.mic_flag:
            function_params += [cgen.Pointer(cgen.POD(type_label, name+"_pointer")) for type_label, name in function_descriptor.value_params]
            return cgen.FunctionDeclaration(cgen.Value(self._pymic_attribute + '\nint', function_descriptor.name),
                                            function_params)
        else:
            function_params += [cgen.POD(type_label, name) for type_label, name in function_descriptor.value_params]
            return cgen.Extern("C", cgen.FunctionDeclaration(cgen.Value('int', function_descriptor.name), function_params))

    def generate_function_body(self, function_descriptor):
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
    """ This class can be used to describe a general C function which receives multiple parameters
    and loops over one of the parameters, executing a given body of statements inside the loop
    """

    """ Pass the name and the body of the function
        Also pass skip_elements, which is an array of the number of elements to skip in each dimension
        while looping
    """
    def __init__(self, name):
        self.name = name
        self.matrix_params = []
        self.value_params = []
        self.local_vars = []

    def add_matrix_param(self, name, shape, dtype, input):
        """ Add a parameter to the function
            Each parameter has an associated name, shape, dtype
        """
        self.matrix_params.append({'name': name, 'shape': shape, 'dtype': dtype, 'input': input})

    def add_value_param(self, name, dtype):
        """ Declare a new value (scalar) param for this function
        Param_type: numpy dtype
        name: name of the param
        """
        self.value_params.append((np.dtype(dtype), name))

    def add_local_variable(self, name, dtype):
        self.local_vars.append((np.dtype(dtype), name))

    def set_body(self, body):
        self.body = body

    @property
    def params(self):
        return self.matrix_params + self.value_params
