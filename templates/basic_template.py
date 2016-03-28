'''
Created on 24 Mar 2016

@author: navjotkukreja
'''
import includes
import cgen


class BasicTemplate(object):
    # Order of names in the following list is important. The resulting code blocks would be placed in the same order as they appear here
    _template_methods = ['includes', 'process_function']

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def includes(self):
        statements = []
        statements += [cgen.Define('M_PI', '3.14159265358979323846')]
        statements += includes.common_include()
        return cgen.Module(statements)

    def generate(self):
        statements = [getattr(self, m)() for m in self._template_methods]
        return cgen.Module(statements)
    
    def process_function(self):
        statements = []
        #statements.append(cgen.Statement("printf(\"element: %d\", input_grid[i1])"))
        statements.append(cgen.Assign("output_grid[i1][i2]", "input_grid[i1][i2] + 3"))
        
        kernel = cgen.Block(statements)
        return self.generate_function("opesci_process", self.dimensions, kernel)
    
    def generate_function(self, name, num_dim, kernel):
        return cgen.FunctionBody(self.generate_function_signature(name, num_dim), self.generate_function_body(num_dim, kernel))

    def generate_function_signature(self, name, num_dim):
        sizes_arr = []
        input_grid = cgen.Value("double", "input_grid")
        output_grid = cgen.Value("double", "output_grid")
        for dim_ind in range(1, num_dim+1):
            size_label = "size"+str(dim_ind)
            sizes_arr.append(cgen.Value("int", size_label))
            input_grid = cgen.Pointer(input_grid)
            output_grid = cgen.Pointer(output_grid)
        function_params = [input_grid, output_grid]+sizes_arr
        return cgen.Extern("C", cgen.FunctionDeclaration(cgen.Value('int', name), function_params))

    def generate_function_body(self, num_dim, kernel):
        statements = []
        #statements.append(cgen.Statement("printf(\"size1: %d, size2: %d\", size1, size2)"))
        body = kernel
        for dim_ind in range(1, num_dim+1):
            dim_var = "i"+str(dim_ind)
            dim_size = "size"+str(dim_ind)
            body = cgen.For(cgen.InlineInitializer(cgen.Value("int", dim_var), 0), dim_var+"<"+dim_size, dim_var+"++", body)
        
        statements.append(body)
        statements.append(cgen.Statement("return 0"))
        return cgen.Block(statements)
