import sympy

from cached_property import cached_property

from devito.tools import filter_ordered, flatten

__all__ = ['Coefficients']


class Coefficients(object):
    """
    Devito class for users to define custom finite difference weights.
    """

    # FIX ME: Instead use __new__ and create specialised classes
    # depending on format of coefficients.
    def __init__(self, *args, **kwargs):

        # FIXME: Add args check.    

        self.data = args
        self.rules = self.rules()

    def rules(self):
        
        # FIXME: Move this
        def generate_rules(d):
            
            deriv_order = d[0]
            function = d[1]
            dim = d[2]
            coeffs = d[-1]
            
            fd_order = len(coeffs)-1
            
            subs = {}
            
            if fd_order == 1:
                indices = [dim, dim + dim.spacing]
            else:
                indices = [(dim + i * dim.spacing) for i in range(-fd_order//2, fd_order//2 + 1)]
            
            for j in range(len(coeffs)):
                W = sympy.Function('W')
                W = W(indices[j], deriv_order, function, dim)
                subs.update({W: coeffs[j]})
        
            return subs
        
        # Figure out when symbolic coefficients can be replaced
        # with user provided coefficients and, if possible, generate
        # replacement rules
        rules = {}
        for d in self.data:
            if isinstance(d, tuple):
                rules.update(generate_rules(d))
                
        print(rules)
                
        return rules
    
    
############################################################################
        
        #print(self)
        #print(d)
        
        #"""
        #Test stuff
        #"""
        #input = filter_sorted(flatten(e.reads for e in expressions))
        #output = filter_sorted(flatten(e.writes for e in expressions))
        ##self.dimensions = filter_sorted(flatten(e.dimensions for e in expressions))
        
        ##print(input)
        
        #function = output[0]
        
        ##print(function)
        
        #dim = function.dimensions
        
        ##print(dim[1])
        
        #dimensions = function.indices
        #space_fd_order = function.space_order
        #time_fd_order = function.time_order if function.is_TimeFunction else 0
        
        
        ## hakz
        ##dim = dim[1]
        ##fd_order = space_fd_order
        
        ## replace
        #for j in range(len(args)):
            
            #def fd_substitutions(expressions, subs):
                #processed = []
                #for e in expressions:
                    #mapper = subs.copy()
                    #processed.append(e.xreplace(mapper))
                #return processed
            
            
            #arg = args[j]
            #dim = arg[0]
            #coeffs = arg[1]
            #fd_order = len(coeffs)-1
            #indices = [(dim + i * dim.spacing) for i in range(-fd_order//2, fd_order//2 + 1)]
            #if fd_order == 1:
                #indices = [dim, dim + dim.spacing]
            ##print(indices)
            #for k in range(len(coeffs)):
                #W = sympy.Function('W')
                #W = W(indices[k])
                ##print(W)
                #subs = {W: coeffs[k]}
                ##for e in expressions:
                    ##print(e)
                #expressions = fd_substitutions(expressions, subs)
                ##expressions = expressions.xreplace({W: coeffs[k]})
                
            ##print(expressions)
        
        #return expressions
