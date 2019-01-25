import sympy
import numpy as np

from devito.finite_differences import generate_indices

__all__ = ['Coefficients', 'Coefficient', 'default_rules']


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

        if any(not arg.is_Coefficient for arg in args):
            raise TypeError("Non Coefficient object within input")

        self.data = args
        self.function_list = self.function_list()
        self.rules = self.rules()

    def function_list(self):
        function_list = ()
        for d in self.data:
            function_list += (d.function,)
        return list(set(function_list))

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
                subs.update({function.fd_coeff_symbol()
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


class Coefficient(object):
    """
    Prepare custom FD coefficients to pass to Coefficients object.

    Parameters
    ----------
    deriv_order : represents the order of the derivative being taken.
    function : represents the function for which the supplied coefficients
               will be used.
    dimension : represents the dimension with respect to which the
                derivative is being taken.
    coefficients : represents the set of finite difference coefficients
                  intended to be used in place of the standard
                  coefficients (obtained from a Taylor expansion).
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

        self.is_Coefficient = True

        self.deriv_order = deriv_order
        self.function = function
        self.dimension = dimension
        self.coefficients = coefficients

    def check_input(self, deriv_order, function, dimension, coefficients):
        if not isinstance(deriv_order, int):
            raise TypeError("Derivative order must be an integer")
        if not function.is_Function:
            raise TypeError("Coefficients must be attached to a valid function")
        if not dimension.is_Dimension:
            raise TypeError("Coefficients must be attached to a valid dimension")
        # Currently only numpy arrays are accepted here.
        # Functionality will be expanded in the near future.
        if not isinstance(coefficients, np.ndarray):
            raise NotImplementedError
        return


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
            subs.update({function.fd_coeff_symbol()
                         (indices[j], deriv_order, function, dim): coeffs[j]})

        return subs

    rules = {}

    # Determine which 'rules' are missing
    sym = get_sym(functions)
    terms = obj.find(sym)
    # FIXME: Unnecessary conversions between lists and sets
    args_present = []
    for term in terms:
        args = term.args
        args_present += [args[1:], ]
    args_present = list(set(args_present))

    coeffs = obj.coefficients
    args_provided = []
    if coeffs:
        for coeff in coeffs.data:
            args_provided += [(coeff.deriv_order, coeff.function, coeff.dimension), ]
    # NOTE: Do we want to throw a warning if the same arg has
    # been provided twice?
    args_provided = list(set(args_provided))
    not_provided = list(set(args_provided).symmetric_difference(set(args_present)))

    if not_provided:
        for i in not_provided:
            rules = {**rules, **generate_subs(i)}

    return rules


def get_sym(functions):
    for j in range(0, len(functions)):
        try:
            sym = functions[j].fd_coeff_symbol()
            return sym
        except:
            pass
    # Shouldn't arrive here
    raise TypeError("Failed to retreive symbol")
