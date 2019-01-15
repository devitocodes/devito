import sympy
import numpy as np

from cached_property import cached_property

from devito.tools import filter_ordered, flatten
from devito.finite_differences import generate_indices

__all__ = ['Coefficients', 'default_rules']


class Coefficients(object):
    """
    Devito class for users to define custom finite difference weights.
    Input must be given as tuple(s) with the following arguments:
    tuple = (deriv_order, function, dimension, coefficients)

    Here,
    deriv_order: represents the order of the derivative being taken.
    function: represents the function for which the supplied coefficients
              will be used.
    dimension: represents the dimension with respect to which the
               derivative is being taken.
    coefficients: represents the set of finite difference coefficients
                  intended to be used in place of the standard
                  coefficients (obtained from a Taylor expansion).

    Coefficient objects created in this manner must then be
    passed to Devito equation objects for the replacement rules
    to take effect.
    """

    def __init__(self, *args, **kwargs):

        self.check_args(args, kwargs)

        self.function_list = self.function_list(args, kwargs)

        self.data = args
        self.rules = self.rules()

    def check_args(self, args, kwargs):
        for arg in args:
            if not isinstance(arg, tuple):
                raise TypeError("Input must be of type tuple.")
            if not isinstance(arg[0], int):
                raise TypeError("Derivative order must be an integer.")
            if not arg[1].is_Function:
                raise TypeError("Coefficients must be attached to a valid function.")
            if not arg[2].is_Dimension:
                raise TypeError("Coefficients must be attached to a valid dimension.")
            # Currently only numpy arrays are accepted here.
            # Functionality will be expanded in the near future.
            if not isinstance(arg[3], np.ndarray):
                raise NotImplementedError
        return

    def function_list(self, args, kwargs):
        function_list = ()
        for arg in args:
            function_list += (arg[1],)
        return list(set(function_list))

    def rules(self):

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

        subs = {}

        indices = generate_indices(dim, dim.spacing, fd_order)

        coeffs = sympy.finite_diff_weights(deriv_order, indices, dim)[-1][-1]

        for j in range(len(coeffs)):
            subs.update({function.fd_coeff_symbol()(indices[j], deriv_order, function, dim): coeffs[j]})

        return subs

    rules = {}

    # Determine which 'rules' are missing
    # FIXME: Manipulating coefficient arrays is potentially dangerous
    # and this should probably be done via subojects.
    sym = functions[0].fd_coeff_symbol()
    terms = obj.find(sym)
    #FIXME: Unnecessary conversions between lists and sets
    args_present = []
    for term in terms:
        args = term.args
        args_present += [args[1:],]
    args_present = list(set(args_present))

    coeffs = obj._coefficients
    args_provided = []
    if coeffs:
        for coeff in coeffs.data:
            args_provided += [coeff[:-1],]
    # NOTE: Do we want to throw a warning if the same arg has
    # been provided twice?
    args_provided = list(set(args_provided))
    not_provided = list(set(args_provided).symmetric_difference(set(args_present)))

    if not_provided:
        for i in not_provided:
            rules = {**rules, **generate_subs(i)}

    return rules
