import sympy
import numpy as np
from cached_property import cached_property

from devito.finite_differences import generate_indices
from devito.tools import filter_ordered, as_tuple
from devito.symbolics.search import retrieve_dimensions

__all__ = ['Coefficient', 'Substitutions', 'default_rules']


class Coefficient(object):
    """
    Prepare custom coefficients to pass to a Substitutions object.

    Parameters
    ----------
    deriv_order : int
        The order of the derivative being taken.
    function : Function
        The function for which the supplied coefficients
        will be used.
    dimension : Dimension
        The dimension with respect to which the
        derivative is being taken.
    weights : np.ndarray
        The set of finite difference weights
        intended to be used in place of the standard
        weights (obtained from a Taylor expansion).

    Examples
    --------
    >>> import numpy as np
    >>> from devito import Grid, Function, Coefficient
    >>> grid = Grid(shape=(4, 4))
    >>> u = Function(name='u', grid=grid, space_order=2, coefficients='symbolic')
    >>> x, y = grid.dimensions

    Now define some partial d/dx FD coefficients of the Function u:

    >>> u_x_coeffs = Coefficient(1, u, x, np.array([-0.6, 0.1, 0.6]))

    And some partial d^2/dy^2 FD coefficients:

    >>> u_y2_coeffs = Coefficient(2, u, y, np.array([0.0, 0.0, 0.0]))

    """

    def __init__(self, deriv_order, function, dimension, weights):

        self._check_input(deriv_order, function, dimension, weights)

        # Ensure the given set of weights is the correct length
        try:
            wl = weights.shape[-1]-1
        except AttributeError:
            wl = len(weights)-1
        if dimension.is_Time:
            if wl != function.time_order:
                raise ValueError("Number of FD weights provided does not "
                                 "match the functions space_order")
        elif dimension.is_Space:
            if wl != function.space_order:
                raise ValueError("Number of FD weights provided does not "
                                 "match the functions space_order")

        self._deriv_order = deriv_order
        self._function = function
        self._dimension = dimension
        self._weights = weights

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
    def index(self):
        """
        The dimension to which the coefficients will be applied plus the offset
        in that dimension if the function is staggered.
        """
        return self._function.indices_ref[self._dimension]

    @property
    def weights(self):
        """The set of weights."""
        return self._weights

    def _check_input(self, deriv_order, function, dimension, weights):
        if not isinstance(deriv_order, int):
            raise TypeError("Derivative order must be an integer")
        try:
            if not function.is_Function:
                raise TypeError("Object is not of type Function")
        except AttributeError:
            raise TypeError("Object is not of type Function")
        try:
            if not dimension.is_Dimension:
                raise TypeError("Coefficients must be attached to a valid dimension")
        except AttributeError:
            raise TypeError("Coefficients must be attached to a valid dimension")
        try:
            weights.is_Function is True
        except AttributeError:
            if not isinstance(weights, np.ndarray):
                raise TypeError("Weights must be of type np.ndarray or a Devito Function")
        return


class Substitutions(object):
    """
    Devito class to convert Coefficient objects into replacent rules
    to be applied when constructing a Devito Eq.

    Examples
    --------
    >>> from devito import Grid, TimeFunction, Coefficient
    >>> grid = Grid(shape=(4, 4))
    >>> u = TimeFunction(name='u', grid=grid, space_order=2, coefficients='symbolic')
    >>> x, y = grid.dimensions

    Now define some partial d/dx FD coefficients of the Function u:

    >>> u_x_coeffs = Coefficient(1, u, x, np.array([-0.6, 0.1, 0.6]))

    And now create our Substitutions object to pass to equation:

    >>> from devito import Substitutions
    >>> subs = Substitutions(u_x_coeffs)

    Now create a Devito equation and pass to it 'subs'

    >>> from devito import Eq
    >>> eq = Eq(u.dt+u.dx, coefficients=subs)

    When evaluated, the derivatives will use the custom coefficients. We can
    check that by

    >>> eq.evaluate
    Eq(0.1*u(t, x, y) - 0.6*u(t, x - h_x, y) + 0.6*u(t, x + h_x, y) \
- u(t, x, y)/dt + u(t + dt, x, y)/dt, 0)

    Notes
    -----
    If a Function is declared with 'symbolic' coefficients and no
    replacement rules for any derivative appearing in a Devito equation,
    the coefficients will revert to those of the 'default' Taylor
    expansion.
    """

    def __init__(self, *args):

        if any(not isinstance(arg, Coefficient) for arg in args):
            raise TypeError("Non Coefficient object within input")

        self._coefficients = args
        self._function_list = self.function_list
        self._rules = self.rules

    @property
    def coefficients(self):
        """The Coefficient objects passed."""
        return self._coefficients

    @cached_property
    def function_list(self):
        return filter_ordered((i.function for i in self.coefficients), lambda i: i.name)

    @cached_property
    def rules(self):

        def generate_subs(i):

            deriv_order = i.deriv_order
            function = i.function
            dim = i.dimension
            index = i.index
            weights = i.weights
            print(deriv_order, function, dim, index, weights)

            if isinstance(weights, np.ndarray):
                fd_order = len(weights)-1
            else:
                fd_order = weights.shape[-1]-1

            subs = {}

            indices, x0 = generate_indices(function, dim, fd_order, side=None)

            # NOTE: This implementation currently assumes that indices are ordered
            # according to their position in the FD stencil. This may not be the
            # case in all schemes and should be changed such that the weights are
            # passed as a dictionary of the form {pos: w} (or something similar).
            if isinstance(weights, np.ndarray):
                for j in range(len(weights)):
                    # Index here is incorrect. Need to pull it from elsewhere
                    subs.update({function._coeff_symbol
                                 (indices[j], deriv_order,
                                  function, index): weights[j]})
            else:
                shape = weights.shape
                x = weights.dimensions
                for j in range(shape[-1]):
                    idx = list(x)
                    idx[-1] = j
                    subs.update({function._coeff_symbol
                                 (indices[j], deriv_order, function, index):
                                     weights[as_tuple(idx)]})

            return subs

        # Figure out when symbolic coefficients can be replaced
        # with user provided coefficients and, if possible, generate
        # replacement rules
        rules = {}
        for i in self.coefficients:
            rules.update(generate_subs(i))
        return rules

    def update_rules(self, obj):
        """Update the specified rules to reflect staggering in an equation"""
        # Determine which 'rules' are expected
        sym = get_sym(self._function_list)
        terms = obj.find(sym)
        args_expected = filter_ordered(term.args[1:] for term in terms)
        args_expected_dim = [(arg[0], arg[1], retrieve_dimensions(arg[2])[0])
                             for arg in args_expected]

        # Modify dictionary keys where expected index does not match index in rules
        rules = self._rules.copy()  # Get a copy to modify, to preserve base rules
        for rule in self._rules:
            rule_arg = rule.args[1:]
            rule_arg_dim = (rule_arg[0], rule_arg[1],
                            retrieve_dimensions(rule_arg[2])[0])
            if rule_arg_dim in args_expected_dim and rule_arg not in args_expected:
                # Rule matches expected in terms of dimensions, but index is
                # mismatched (due to staggering of equation)

                # Find index in args_expected_dim
                pos = args_expected_dim.index(rule_arg_dim)
                # Replace key in rules with one using modified index taken from
                # the expected
                replacement = rule.args[:-1] + (args_expected[pos][-1],)
                rules[sym(*replacement)] = rules.pop(rule)

        return rules


def default_rules(obj, functions):

    def generate_subs(deriv_order, function, index):
        dim = retrieve_dimensions(index)[0]

        if dim.is_Time:
            fd_order = function.time_order
        elif dim.is_Space:
            fd_order = function.space_order
        else:
            # Shouldn't arrive here
            raise TypeError("Dimension type not recognised")

        subs = {}

        mapper = {dim: index}

        indices, x0 = generate_indices(function, dim,
                                       fd_order, side=None, x0=mapper)

        coeffs = sympy.finite_diff_weights(deriv_order, indices, x0)[-1][-1]

        for j in range(len(coeffs)):
            subs.update({function._coeff_symbol
                        (indices[j], deriv_order, function, index): coeffs[j]})

        return subs

    # Determine which 'rules' are missing
    print("Functions", functions)
    sym = get_sym(functions)
    print("Sym", sym)
    terms = obj.find(sym)
    print("Terms", terms)
    args_present = filter_ordered(term.args[1:] for term in terms)

    subs = obj.substitutions
    if subs:
        args_provided = [(i.deriv_order, i.function, i.index)
                         for i in subs.coefficients]
    else:
        args_provided = []

    # NOTE: Do we want to throw a warning if the same arg has
    # been provided twice?
    args_provided = list(set(args_provided))
    print("Args present", args_present)
    print("Args provided", args_provided)

    # Need to compare dimensions,, not indices, then take the index from
    # args_present to pass to generate_subs()
    args_pres_dim = [(arg[0], arg[1], retrieve_dimensions(arg[2])[0])
                     for arg in args_present]
    args_prov_dim = [(arg[0], arg[1], retrieve_dimensions(arg[2])[0])
                     for arg in args_provided]

    not_provided = [args_present[i] for i in range(len(args_present))
                    if args_pres_dim[i] not in frozenset(args_prov_dim)]

    rules = {}
    for i in not_provided:
        rules = {**rules, **generate_subs(*i)}

    return rules


def get_sym(functions):
    for f in functions:
        try:
            sym = f._coeff_symbol
            return sym
        except AttributeError:
            pass
    # Shouldn't arrive here
    raise TypeError("Failed to retreive symbol")
