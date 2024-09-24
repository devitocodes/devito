from functools import cached_property
from warnings import warn


class DevitoDeprecation():

    @cached_property
    def coeff_warn(self):
        warn("The Coefficient API is deprecated and will be removed, coefficients should"
             "be passed directly to the derivative object `u.dx(weights=...)",
             DeprecationWarning, stacklevel=2)
        return

    @cached_property
    def symbolic_warn(self):
        warn("coefficients='symbolic' is deprecated, coefficients should"
             "be passed directly to the derivative object `u.dx(weights=...)",
             DeprecationWarning, stacklevel=2)
        return


deprecations = DevitoDeprecation()
