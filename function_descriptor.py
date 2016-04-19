import numpy as np


class FunctionDescriptor(object):
    """ This class can be used to describe a general C function which receives multiple parameters
    and loops over one of the parameters, executing a given body of statements inside the loop
    """

    """ Pass the name and the body of the function
        Also pass skip_elements, which is an array of the number of elements to skip in each dimension
        while looping
    """
    def __init__(self, name, body, skip_elements=None):
        self.name = name
        self.body = body
        self.skip_elements = skip_elements
        self.matrix_params = []
        self.value_params = []

    """ Add a parameter to the function
        A function may have any number of parameters but only one may be the looper
        Each parameter has an associated name and shape
    """
    def add_matrix_param(self, name, shape, looper=False):
        if looper is True:
            assert(self.get_looper_matrix() is None)
            assert((self.skip_elements is None) or (len(shape) == len(self.skip_elements)))
            self._loop_direction = (False,)*len(shape)
        self.matrix_params.append({'name': name, 'shape': shape, 'looper': looper})

    """ Get the parameter of the function which is the looper, i.e. it defines the dimensions over which the primary loop
        of the function is run
    """
    def get_looper_matrix(self):
        for param in self.matrix_params:
            if param['looper'] is True:
                return param
        return None

    """ Declare a new value (scalar) param for this function
        Param_type: numpy dtype
        name: name of the param
    """
    def add_value_param(self, param_type, name):
        self.value_params.append((np.dtype(param_type), name))

    @property
    def params(self):
        return self.matrix_params + self.value_params

    @property
    def loop_direction(self):
        return self._loop_direction

    @loop_direction.setter
    def loop_direction(self, loop_direction):
        looper = self.get_looper_matrix()
        assert looper is not None
        assert len(loop_direction) == len(looper['shape'])
        self._loop_direction = loop_direction
