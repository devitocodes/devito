import cgen_wrapper as cgen
import numpy as np


class FunctionManager(object):
    """Class that accepts a list of FunctionDescriptor objects and generates the C
        function represented by it
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

            # sets function return type if specified
            return_type = function_descriptor.return_type if function_descriptor.return_type else self._pymic_attribute + '\nint'

            return cgen.FunctionDeclaration(cgen.Value(return_type, function_descriptor.name),
                                            function_params)
        else:
            function_params += [cgen.POD(type_label, name) for type_label, name in function_descriptor.value_params]

            # used in auto tune
            if function_descriptor.return_type:  # sets return type if specified
                return cgen.FunctionDeclaration(cgen.Value(function_descriptor.return_type, function_descriptor.name),
                                                function_params)

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

    # appends main at function to the list function desc. Works with assumption that we are auto tuning first function.
    # param - list<FunctionDescriptor> function_descriptors - list to which main function is appended
    # param - Propagator propagator - propagator. Used for retrieving at markers
    def append_main_at_function(self, function_descriptors, propagator):
        statements = []  # statements for cgen.block
        pnames = []  # parameter names
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
            allocation_str = "%s* %s = (%s*)malloc(%ssizeof(%s))" % (ptype, pname, ptype, array_size_str, ptype)
            statements.append(cgen.Statement(allocation_str))

        statements.append(cgen.Pragma("isat marker %s" % propagator.at_markers[1][0]))  # tells at measure start

        #                      cuts the [    removes ]        removes ' symbol
        function_args_str = str(pnames)[1:].replace(']', '').replace('\'', '')

        # call to function that we are auto tuning
        statements.append(cgen.Statement("%s(%s)" % (function_descriptors[0].name, function_args_str)))

        statements.append(cgen.Pragma("isat marker %s" % propagator.at_markers[1][1]))  # tells at measure end

        main_fd.set_body(cgen.Block(statements))  # set whole function body
        function_descriptors.append(main_fd)  # append function descriptor


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
        self.return_type = None

    # adds parameter in function call/assigns values to it in form(dtype (*name)[shape]=(dtype (*)[shape]) name+_vec;)
    # ex function_name(double *m){
    #    double (*m)[180][180] = (double (*)[180][180]) m_vec;
    def add_matrix_param(self, name, shape, dtype):
        """ Add a parameter to the function
            Each parameter has an associated name, shape, dtype
        """
        self.matrix_params.append({'name': name, 'shape': shape, 'dtype': dtype})

    def add_value_param(self, name, dtype):
        """ Declare a new value (scalar) param for this function
        Param_type: numpy dtype
        name: name of the param
        """
        self.value_params.append((np.dtype(dtype), name))

    def add_local_variable(self, name, dtype):
        try:
            self.local_vars.append((np.dtype(dtype), name))
        except:
            self.local_vars.append((dtype, name))

    # Body has to be a single cgen.Statement or block of statements as cgen.Block.
    # otherwise function manager complains
    def set_body(self, body):
        self.body = body

    @property
    def params(self):
        return self.matrix_params + self.value_params
