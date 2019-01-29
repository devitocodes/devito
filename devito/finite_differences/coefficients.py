import sympy
import numpy as np
from cached_property import cached_property

from devito.finite_differences import generate_indices
from devito.tools import filter_ordered

__all__ = ['Coefficients', 'Coefficient', 'default_rules']


class Coefficient(object):
    """
    Prepare custom FD coefficients to pass to Coefficients object.

    Parameters
    ----------
    deriv_order : integer
        Represents the order of the derivative being taken.
    function : Function
        Represents the function for which the supplied coefficients
        will be used.
    dimension : Dimension
        Represents the dimension with respect to which the
        derivative is being taken.
    coefficients : np.ndarray
        Represents the set of finite difference coefficients
        intended to be used in place of the standard
        coefficients (obtained from a Taylor expansion).

    Example
    -------
    >>> from devito import Grid, Function, Coefficient
    >>> grid = Grid(shape=(4, 4))
    >>> u = Function(name='u', grid=grid, space_order=2, coefficients='symbolic')
    >>> x, y = grid.dimensions

    Now define some partial d/dx FD coefficients of the Function u:

    >>> u_x_coeffs = Coefficient(1, u, x, np.array([-0.6, 0.1, 0.6]))

    And some partial d^2/dy^2 FD coefficients:

    >>> u_y2_coeffs = Coefficient(2, u, y, np.array([0.0, 0.0, 0.0]))

    """

    def __init__(self, deriv_order, function, dimension, coefficients):

        self.check_input(deriv_order, function, dimension, coefficients)

        # Ensure the given set of coefficients is the correct length
        if dimension.is_Time:
            if len(coefficients)-1 != function.time_order:
                raise ValueError("Number FD weights provided does not "
                                 "match the functions space_order")
        elif dimension.is_Space:
            if len(coefficients)-1 != function.space_order:
                raise ValueError("Number FD weights provided does not "
                                 "match the functions space_order")

        self._deriv_order = deriv_order
        self._function = function
        self._dimension = dimension
        self._coefficients = coefficients

    @property
    def deriv_order(self):
        """The derivative order."""
        return self._deriv_order

    @property
    def function(self):
        """The function to which the coefficients belong."""
        return self._function

    @property
    def dimension(self):
        """The dimension to which the coefficients will be applied."""
        return self._dimension

    @property
    def coefficients(self):
        """The set of coefficients."""
        return self._coefficients

    def check_input(self, deriv_order, function, dimension, coefficients):
        if not isinstance(deriv_order, int):
            raise TypeError("Derivative order must be an integer")
        # NOTE: Can potentially be tidied up following the implementation
        # of lazy evaluation.
        try:
            if not function.is_Function:
                raise TypeError("Object is not of type Function")
        except:
            raise TypeError("Object is not of type Function")
        try:
            if not dimension.is_Dimension:
                raise TypeError("Coefficients must be attached to a valid dimension")
        except:
            raise TypeError("Coefficients must be attached to a valid dimension")
        # Currently only numpy arrays are accepted here.
        # Functionality will be expanded in the near future.
        if not isinstance(coefficients, np.ndarray):
            raise NotImplementedError
        return


# FIXME: Class naming a bit unclear
class Coefficients(object):
    """
    Devito class for users to define custom finite difference weights.
    Input must be given as Devito Coefficient objects.

    Coefficients objects created in this manner must then be
    passed to Devito equation objects for the replacement rules
    to take effect.
    """

    def __init__(self, *args):

        if any(not isinstance(arg, Coefficient) for arg in args):
            raise TypeError("Non Coefficient object within input")

        self._data = args
        self._function_list = self.function_list
        self._rules = self.rules

    @property
    def data(self):
        """The Coefficient objects passed."""
        return self._data

    @cached_property
    def function_list(self):
        return list(set([d.function for d in self.data]))

    @cached_property
    def rules(self):

        def generate_subs(d):

            deriv_order = d.deriv_order
            function = d.function
            dim = d.dimension
            coeffs = d.coefficients

            fd_order = len(coeffs)-1

            side = function.get_side(dim, deriv_order)
            stagger = function.get_stagger(dim, deriv_order)

            subs = {}

            indices, x0 = generate_indices(function, dim, dim.spacing, fd_order,
                                           side=side, stagger=stagger)

            for j in range(len(coeffs)):
                subs.update({function.fd_coeff_symbol
                             (indices[j], deriv_order, function, dim): coeffs[j]})

            return subs

        # Figure out when symbolic coefficients can be replaced
        # with user provided coefficients and, if possible, generate
        # replacement rules
        rules = {}
        for d in self.data:
            if isinstance(d.coefficients, np.ndarray):
                rules.update(generate_subs(d))

        return rules


def default_rules(obj, functions):

    def generate_subs(d):

        deriv_order = d[0]
        function = d[1]
        dim = d[2]

        if dim.is_Time:
            fd_order = function.time_order
        elif dim.is_Space:
            fd_order = function.space_order
        else:
            # Shouldn't arrive here
            raise TypeError("Dimension type not recognised")

        side = function.get_side(dim, deriv_order)
        stagger = function.get_stagger(dim, deriv_order)

        subs = {}

        indices, x0 = generate_indices(function, dim, dim.spacing, fd_order,
                                       side=side, stagger=stagger)

        coeffs = sympy.finite_diff_weights(deriv_order, indices, x0)[-1][-1]

        for j in range(len(coeffs)):
            subs.update({function.fd_coeff_symbol
                         (indices[j], deriv_order, function, dim): coeffs[j]})

        return subs

    # Determine which 'rules' are missing
    sym = get_sym(functions)
    terms = obj.find(sym)
    args_present = filter_ordered(term.args[1:] for term in terms)

    coeffs = obj.coefficients
    if coeffs:
        args_provided = [(coeff.deriv_order, coeff.function, coeff.dimension)
                         for coeff in coeffs.data]
    else:
        args_provided = []

    # NOTE: Do we want to throw a warning if the same arg has
    # been provided twice?
    args_provided = list(set(args_provided))
    not_provided = list(set(args_provided).symmetric_difference(set(args_present)))

    rules = {}
    if not_provided:
        for i in not_provided:
            rules = {**rules, **generate_subs(i)}

    return rules


def get_sym(functions):
    for j in range(0, len(functions)):
        try:
            sym = functions[j].fd_coeff_symbol
            return sym
        except:
            pass
    # Shouldn't arrive here
    raise TypeError("Failed to retreive symbol")
