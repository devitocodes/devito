"""User API to specify equations."""

import sympy

__all__ = ['Eq', 'Inc', 'solve']


class Eq(sympy.Eq):

    """
    An equal relation between two objects, the left-hand side and the
    right-hand side.
    The left-hand side may be a Function or a SparseFunction. The right-hand
    side may be any arbitrary expressions with numbers, Dimensions, Constants,
    Functions and SparseFunctions as operands.
    Parameters
    ----------
    lhs : Function or SparseFunction
        The left-hand side.
    rhs : expr-like
        The right-hand side.
    subdomain : SubDomain, optional
        To restrict the computation of the Eq to a particular sub-region in the
        computational domain.
    coefficients : Coefficients, optional
        Can be used to replace symbolic finite difference weights with user defined weights.
    Examples
    --------
    >>> from devito import Grid, Function, Eq
    >>> grid = Grid(shape=(4, 4))
    >>> f = Function(name='f', grid=grid)
    >>> Eq(f, f + 1)
    Eq(f(x, y), f(x, y) + 1)
    Any SymPy expressions may be used in the right-hand side.
    >>> from sympy import sin
    >>> Eq(f, sin(f.dx)**2)
    Eq(f(x, y), sin(f(x, y)/h_x - f(x + h_x, y)/h_x)**2)
    Notes
    -----
    An Eq can be thought of as an assignment in an imperative programming language
    (e.g., ``a[i] = b[i]*c``).
    """

    is_Increment = False

    def __new__(cls, *args, **kwargs):
        kwargs['evaluate'] = False
        subdomain = kwargs.pop('subdomain', None)
        coefficients = kwargs.pop('coefficients', None)
        obj = sympy.Eq.__new__(cls, *args, **kwargs)
        obj._subdomain = subdomain
        obj._coefficients = coefficients
        return obj

    @property
    def subdomain(self):
        """The SubDomain in which the Eq is defined."""
        return self._subdomain

    @property
    def coefficients(self):
        return self._coefficients

    def xreplace(self, rules):
        """"""
        return self.func(self.lhs.xreplace(rules), self.rhs.xreplace(rules),
                         subdomain=self._subdomain)


class Inc(Eq):

    """
    An increment relation between two objects, the left-hand side and the
    right-hand side.

    Examples
    --------
    Inc may be used to express tensor contractions. Below, a summation along
    the user-defined Dimension ``i``.

    >>> from devito import Grid, Dimension, Function, Inc
    >>> grid = Grid(shape=(4, 4))
    >>> x, y = grid.dimensions
    >>> i = Dimension(name='i')
    >>> f = Function(name='f', grid=grid)
    >>> g = Function(name='g', shape=(10, 4, 4), dimensions=(i, x, y))
    >>> Inc(f, g)
    Inc(f(x, y), g(i, x, y))

    Notes
    -----
    An Inc can be thought of as the augmented assignment '+=' in an imperative
    programming language (e.g., ``a[i] += c``).
    """

    is_Increment = True

    def __str__(self):
        return "Inc(%s, %s)" % (self.lhs, self.rhs)

    __repr__ = __str__


def solve(eq, target, **kwargs):
    """
    Algebraically rearrange an Eq w.r.t. a given symbol.

    This is a wrapper around ``sympy.solve``.

    Parameters
    ----------
    eq : expr-like
        The equation to be rearranged.
    target : symbol
        The symbol w.r.t. which the equation is rearranged. May be a `Function`
        or any other symbolic object.
    **kwargs
        Symbolic optimizations applied while rearranging the equation. For more
        information. refer to ``sympy.solve.__doc__``.
    """
    # Enforce certain parameters to values that are known to guarantee a quick
    # turnaround time
    kwargs['rational'] = False  # Avoid float indices
    kwargs['simplify'] = False  # Do not attempt premature optimisation
    return sympy.solve(eq, target, **kwargs)[0]


    def _apply_fd_coefficients(self, expressions, args):
        """
        Test stuff
        """
        input = filter_sorted(flatten(e.reads for e in expressions))
        output = filter_sorted(flatten(e.writes for e in expressions))
        #self.dimensions = filter_sorted(flatten(e.dimensions for e in expressions))
        
        #print(input)
        
        function = output[0]
        
        #print(function)
        
        dim = function.dimensions
        
        #print(dim[1])
        
        dimensions = function.indices
        space_fd_order = function.space_order
        time_fd_order = function.time_order if function.is_TimeFunction else 0
        
        
        # hakz
        #dim = dim[1]
        #fd_order = space_fd_order
        
        # replace
        for j in range(len(args)):
            
            def fd_substitutions(expressions, subs):
                processed = []
                for e in expressions:
                    mapper = subs.copy()
                    processed.append(e.xreplace(mapper))
                return processed
            
            
            arg = args[j]
            dim = arg[0]
            coeffs = arg[1]
            fd_order = len(coeffs)-1
            indices = [(dim + i * dim.spacing) for i in range(-fd_order//2, fd_order//2 + 1)]
            if fd_order == 1:
                indices = [dim, dim + dim.spacing]
            #print(indices)
            for k in range(len(coeffs)):
                W = sympy.Function('W')
                W = W(indices[k])
                #print(W)
                subs = {W: coeffs[k]}
                #for e in expressions:
                    #print(e)
                expressions = fd_substitutions(expressions, subs)
                #expressions = expressions.xreplace({W: coeffs[k]})
                
            #print(expressions)
        
        return expressions
