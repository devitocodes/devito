import sympy
import numpy as np

from cached_property import cached_property

from devito.tools import filter_ordered, flatten

__all__ = ['Coefficients']


class Coefficients(object):
    """
    Devito class for users to define custom finite difference weights.
    """
 
    def __init__(self, *args, **kwargs):

        self.check_args(args, kwargs)

        self.data = args
        self.rules = self.rules()
        
    def check_args(self, args, kwargs):
        for arg in args:
            if isinstance(arg, tuple):
                assert isinstance(arg[0], int)
                assert(arg[1].is_Function)
                assert(arg[2].is_Dimension)
                assert isinstance(arg[3], np.ndarray)
            else:
                raise NotImplementedError
        return

    def rules(self):
        
        # FIXME: Move this?
        def generate_subs(d):
            
            deriv_order = d[0]
            function = d[1]
            dim = d[2]
            coeffs = d[-1]
            
            fd_order = len(coeffs)-1
            
            subs = {}
            
            # FIXME: indices should not be regenerated here?
            if fd_order == 1:
                indices = [dim, dim + dim.spacing]
            else:
                indices = [(dim + i * dim.spacing) for i in range(-fd_order//2, fd_order//2 + 1)]
            
            for j in range(len(coeffs)):
                subs.update({function.fd_coeff_symbol()(indices[j], deriv_order, function, dim): coeffs[j]})
        
            return subs
        
        # Figure out when symbolic coefficients can be replaced
        # with user provided coefficients and, if possible, generate
        # replacement rules
        rules = {}
        for d in self.data:
            if isinstance(d, tuple):
                rules.update(generate_subs(d))
                
        return rules
