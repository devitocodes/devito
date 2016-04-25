import cgen_wrapper as cgen
import function_descriptor


class FunctionManager(object):
    """Class that accepts a list of FunctionDescriptor objects and generates the C 
        function represented by it
    """
    libraries = ['cassert', 'cstdlib', 'cmath', 'iostream',
                 'fstream', 'vector', 'cstdio', 'string', 'inttypes.h']

    def __init__(self, function_descriptors):
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
        for param in function_descriptor.matrix_params:
            param_vec_def = cgen.Pointer(cgen.POD(param['dtype'], param['name']+"_vec"))
            sizes_arr = [cgen.Value("int", param['name']+"_size"+str(dim_ind+1)) for dim_ind in range(param['num_dim'])]
            function_params += [param_vec_def] + sizes_arr
        
        for group_name, group_info in function_descriptor.var_groups.iteritems():
            var_defs = [cgen.Pointer(cgen.POD(group_info['dtype'], name+"_vec")) for name in group_info['vars']]
            sizes = [cgen.Value("int", group_name+"_size"+str(dim_ind+1)) for dim_ind in range(group_info['num_dim'])]
            function_params += var_defs + sizes    
            
        function_params += [cgen.POD(type_label, name) for type_label, name in function_descriptor.value_params]
        return cgen.Extern("C",
                           cgen.FunctionDeclaration(cgen.Value('int', function_descriptor.name),
                                                    function_params)
                           )

    def generate_function_body(self, function_descriptor):
        statements = [cgen.POD(var[0], var[1]) for var in function_descriptor.local_vars]
        for param in function_descriptor.matrix_params:
                num_dim = param['num_dim']
                arr = "".join(
                    ["[%s_size%d]" % (param['name'], i)
                     for i in range(num_dim-1, 0, -1)]
                )
                cast_pointer = cgen.Initializer(
                    cgen.POD(param['dtype'], "(*%s)%s" % (param['name'], arr)),
                    '(%s (*)%s) %s' % (cgen.dtype_to_ctype(param['dtype']), arr, param['name']+"_vec")
                )
                statements.append(cast_pointer)

        for group_name, group_info in function_descriptor.var_groups.iteritems():
                num_dim = group_info['num_dim']
                arr = "".join(
                    ["[%s_size%d]" % (group_name, i)
                     for i in range(num_dim-1, 0, -1)]
                )
                for var in group_info['vars']:
                    cast_pointer = cgen.Initializer(
                        cgen.POD(group_info['dtype'], "(*%s)%s" % (var, arr)),
                        '(%s (*)%s) %s' % (cgen.dtype_to_ctype(group_info['dtype']), arr, var+"_vec")
                    )
                    statements.append(cast_pointer)
        statements.append(function_descriptor.body)
        statements.append(cgen.Statement("return 0"))
        return cgen.Block(statements)
