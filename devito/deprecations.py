from functools import cached_property
from warnings import warn


class DevitoDeprecation():

    @cached_property
    def coeff_warn(self):
        warn("The Coefficient API is deprecated and will be removed, coefficients should"
             " be passed directly to the derivative object `u.dx(weights=...)",
             DeprecationWarning, stacklevel=2)
        return

    @cached_property
    def symbolic_warn(self):
        warn("coefficients='symbolic' is deprecated, coefficients should"
             " be passed directly to the derivative object `u.dx(weights=...)",
             DeprecationWarning, stacklevel=2)
        return

    @cached_property
    def subdomain_warn(self):
        warn("Passing `SubDomain`s to `Grid` on instantiation using `mygrid ="
             " Grid(..., subdomains=(mydomain, ...))` is deprecated. The `Grid`"
             " should instead be passed as a kwarg when instantiating a subdomain"
             " `mydomain = MyDomain(grid=mygrid)`",
             DeprecationWarning, stacklevel=2)
        return

    @cached_property
    def constant_factor_warn(self):
        warn("Using a `Constant` as a factor when creating a ConditionalDimension"
             " is deprecated. Use an integer instead.",
             DeprecationWarning, stacklevel=2)
        return


deprecations = DevitoDeprecation()
