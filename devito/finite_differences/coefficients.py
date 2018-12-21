import sympy
import numpy as np

from cached_property import cached_property

from devito.tools import filter_ordered, flatten
from devito.finite_differences import generate_indices

__all__ = ['Coefficients', 'default_rules']


class Coefficients(object):
    """
    Devito class for users to define custom finite difference weights.
    """

    # FIXME: Make interface explicit.
    # Add optional argument nodes and allow
    # coefficients to also be a list of np.ndarray's
    # of size nodes or a 'function'.
    # Then we should be able to generate the default
    # replacement rules during the __init__ is required.
    # (and we can then clean equation.py up a bit)
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
            
            indices = generate_indices(dim, dim.spacing, fd_order)
            
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

# FIXME: Currently broken.
# FIXME: Overall code can be reduced plus fix function calls.
# FIXME: finite_difference.py possibly needs some re-factoring
# FIXME: to make everything 'smoother'.
def default_rules(obj):
    
    # FIXME: Needs modification + re-location
    def generate_subs(d):

        #deriv_order = d[0]
        #function = d[1]
        #dim = d[2]
        #coeffs = d[-1]

        fd_order = len(coeffs)-1

        subs = {}

        indices = generate_indices(dim, dim.spacing, fd_order)
        
        c = finite_diff_weights(deriv_order, indices, dim)[-1][-1]

        for j in range(len(coeffs)):
            subs.update({function.fd_coeff_symbol()(indices[j], deriv_order, function, dim): coeffs[j]})

        return subs

    rules = {}
    return rules
