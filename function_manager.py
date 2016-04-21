import cgen_wrapper as cgen


class FunctionManager(object):
    # Order of names in the following list is important.
    # The resulting code blocks would be placed in the same
    # order as they appear here

    libraries = ['cassert', 'cstdlib', 'cmath', 'iostream',
                 'fstream', 'vector', 'cstdio', 'string', 'inttypes.h']
    def __init__(self, function_descriptors):
        for function_descriptor in function_descriptors:
            assert(function_descriptor.get_looper_matrix() is not None)
        self.function_descriptors = function_descriptors
        self._defines = []

    def includes(self):
        statements = []
        statements += self._defines
        statements += [cgen.Include(s) for s in self.libraries]
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
        for param in function_descriptor.params:
            sizes_arr = []
            try:
                # Assume the parameter is a matrix parameter
                param_vec_def = cgen.Pointer(cgen.Value("double", param['name']+"_vec"))
                num_dim = len(param['shape'])
                for dim_ind in range(num_dim, 0, -1):
                    size_label = param['name']+"_size"+str(dim_ind)
                    sizes_arr.append(cgen.Value("int", size_label))
                function_params = function_params + [param_vec_def] + sizes_arr
            except TypeError:  # Unless it is a value parameter, in which case make sure
                assert(len(param) == 2)

        function_params += [cgen.POD(type_label, name) for type_label, name in function_descriptor.value_params]
        return cgen.Extern("C",
                           cgen.FunctionDeclaration(cgen.Value('int', function_descriptor.name),
                                                    function_params)
                           )

    def generate_function_body(self, function_descriptor):
        statements = []
        for param in function_descriptor.params:
            try:
                num_dim = len(param['shape'])
                arr = "".join(
                    ["[%s_size%d]" % (param['name'], i)
                     for i in range(num_dim-1, 0, -1)]
                )
                cast_pointer = cgen.Initializer(
                    cgen.Value("double", "(*%s)%s" % (param['name'], arr)),
                    '(%s (*)%s) %s' % ("double", arr, param['name']+"_vec")
                )
                statements.append(cast_pointer)
            except TypeError:  # It might be a value parameter
                pass  # Do nothing, in that case
        body = function_descriptor.body
        looper_param = function_descriptor.get_looper_matrix()
        num_dim = len(looper_param['shape'])
        skip_elements = function_descriptor.skip_elements
        for dim_ind in range(1, num_dim+1):
            dim_var = "i"+str(dim_ind)
            if skip_elements is not None:
                skip = skip_elements[dim_ind-1]
            else:
                skip = 0

            dim_size = looper_param['name']+"_size"+str(dim_ind)
            if skip > 0:
                dim_size = dim_size+"-"+str(skip)
            body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var),
                                                   skip),
                            dim_var + "<" + dim_size,
                            dim_var + "++",
                            body
                            )

        statements.append(body)
        statements.append(cgen.Statement("return 0"))
        return cgen.Block(statements)
