# This class can be used to describe a general C function which receives multiple parameters
# and loops over one of the parameters, executing a given body of statements inside the loop
class FunctionDescriptor(object):
    
    # Pass the name and the body of the function
    # Also pass skip_elements, which is an array of the number of elements to skip in each dimension
    # while looping
    def __init__(self, name, body, skip_elements = None):
        self.name = name
        self.body = body
        self.skip_elements = skip_elements
        self.params = []
    
    # Add a parameter to the function
    # A function may have any number of parameters but only one may be the looper
    # Each parameter has an associated name and shape
    def add_param(self, name, shape, looper = False):
        if looper == True:
            assert(self.get_looper() is None)
            assert((self.skip_elements is None) or (len(shape)==len(self.skip_elements)))
        self.params.append({'name':name, 'shape':shape, 'looper':looper})
    
    # Get the parameter of the function which is the looper, i.e. it defines the dimensions over which the primary loop 
    # of the function is run
    def get_looper(self):
        for param in self.params:
            if param['looper']==True:
                return param
        return None
                