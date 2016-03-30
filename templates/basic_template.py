import includes
import cgen


class BasicTemplate(object):
    # Order of names in the following list is important.
    # The resulting code blocks would be placed in the same
    # order as they appear here

    _template_methods = ['includes', 'process_function']

    def __init__(self, dimensions, kernel):
        self.dimensions = dimensions
        self.kernel = kernel

    def includes(self):
        statements = []
        statements += [cgen.Define('M_PI', '3.14159265358979323846')]
        statements += includes.common_include()
        return cgen.Module(statements)

    def generate(self):
        statements = [getattr(self, m)() for m in self._template_methods]
        return cgen.Module(statements)

    def process_function(self):
        return self.generate_function("opesci_process",
                                      self.dimensions, self.kernel)

    def generate_function(self, name, num_dim, kernel):
        return cgen.FunctionBody(self.generate_function_signature(name,
                                                                  num_dim),
                                 self.generate_function_body(num_dim, kernel)
                                 )

    def generate_function_signature(self, name, num_dim):
        sizes_arr = []
        input_grid = cgen.Value("double", "input_grid_vec")
        output_grid = cgen.Value("double", "output_grid_vec")
        for dim_ind in range(num_dim, 0, -1):
            size_label = "size"+str(dim_ind)
            sizes_arr.append(cgen.Value("int", size_label))
        input_grid = cgen.Pointer(input_grid)
        output_grid = cgen.Pointer(output_grid)
        function_params = [input_grid, output_grid]+sizes_arr
        return cgen.Extern("C",
                           cgen.FunctionDeclaration(cgen.Value('int', name),
                                                    function_params)
                           )

    def generate_function_body(self, num_dim, kernel):
        statements = []
        arr = "".join(["[size%d]" % i for i in range(num_dim-1, 0, -1)])
        cast_pointer_in = cgen.Initializer(cgen.Value("double",
                                                      "(*%s)%s" %
                                                      ("input_grid", arr)
                                                      ),
                                           '(%s (*)%s) %s' %
                                           ("double", arr, "input_grid_vec")
                                           )
        cast_pointer_out = cgen.Initializer(cgen.Value("double",
                                                       "(*%s)%s" %
                                                       ("output_grid", arr)
                                                       ),
                                            '(%s (*)%s) %s' %
                                            ("double", arr, "output_grid_vec")
                                            )
        statements.append(cast_pointer_in)
        statements.append(cast_pointer_out)
        body = kernel
        for dim_ind in range(1, num_dim+1):
            dim_var = "i"+str(dim_ind)
            dim_size = "size"+str(dim_ind)
            body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var),
                                                   0),
                            dim_var + "<" + dim_size,
                            dim_var + "++",
                            body
                            )

        statements.append(body)
        statements.append(cgen.Statement("return 0"))
        return cgen.Block(statements)
