from devito.deprecations import deprecations

__all__ = ['Coefficient', 'Substitutions']


class Coefficient:
    def __init__(self, deriv_order, function, dimension, weights):
        deprecations.coeff_warn
        self._weights = weights
        self._deriv_order = deriv_order
        self._function = function
        self._dimension = dimension

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
    def weights(self):
        """The set of weights."""
        return self._weights


class Substitutions:
    def __init__(self, *args):
        deprecations.coeff_warn
        if any(not isinstance(arg, Coefficient) for arg in args):
            raise TypeError("Non Coefficient object within input")

        self._coefficients = args

    @property
    def coefficients(self):
        """The Coefficient objects passed."""
        return self._coefficients
