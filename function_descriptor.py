import numpy as np


class FunctionDescriptor(object):
    """ This class can be used to describe a general C function which receives multiple parameters
    and loops over one of the parameters, executing a given body of statements inside the loop
    """

    """ Pass the name and the body of the function
        Also pass skip_elements, which is an array of the number of elements to skip in each dimension
        while looping
    """
    def __init__(self, name, body):
        self.name = name
        self.body = body
        self.matrix_params = []
        self.value_params = []
        self.local_vars = []
        self.var_groups = {}

    """ Add a parameter to the function
        A function may have any number of parameters but only one may be the looper
        Each parameter has an associated name and shape
    """
    def add_matrix_param(self, name, num_dim, dtype, group = None):
        if group is None:
            self.matrix_params.append({'name': name, 'num_dim': num_dim, 'dtype': dtype})
        else:
            existing_group = self.var_groups.get(group, {'dtype': dtype, 'num_dim': num_dim, 'vars':[]})
            assert(existing_group['dtype'] == dtype)
            assert(existing_group['num_dim'] == num_dim)
            existing_group['vars'].append(name)
            self.var_groups[group] = existing_group

    """ Declare a new value (scalar) param for this function
        Param_type: numpy dtype
        name: name of the param
    """
    def add_value_param(self, name, dtype):
        self.value_params.append((np.dtype(dtype), name))

    def add_local_variable(self, name, dtype):
        self.local_vars.append((np.dtype(dtype), name))

    @property
    def params(self):
        return self.matrix_params + self.value_params
