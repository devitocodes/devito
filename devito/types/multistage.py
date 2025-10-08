from devito.types.equation import Eq
from devito.types.dense import Function
from devito.symbolics import uxreplace
import numpy as np
from devito.types.array import Array
from types import MappingProxyType

method_registry = {}


def register_method(cls=None, *, aliases=None):
    """
    Register a time integration method class.

    Parameters
    ----------
    cls : class, optional
        The method class to register.
    aliases : list of str, optional
        Additional aliases for the method.
    """
    def decorator(cls):
        # Register the class name
        method_registry[cls.__name__] = cls

        # Register any aliases
        if aliases:
            for alias in aliases:
                method_registry[alias] = cls

        return cls

    if cls is None:
        # Called as @register_method(aliases=['alias1'])
        return decorator
    else:
        # Called as @register_method
        return decorator(cls)


def resolve_method(method):
    """
    Resolve a time integration method by name.

    Parameters
    ----------
    method : str
        Name or alias of the time integration method.

    Returns
    -------
    class
        The method class.

    Raises
    ------
    ValueError
        If the method is not found in the registry.
    """
    try:
        return method_registry[method]
    except KeyError:
        available = sorted(method_registry.keys())
        raise ValueError(
            f"The time integrator '{method}' is not implemented. "
            f"Available methods: {available}"
        )


def multistage_method(lhs, rhs, method, degree=None, source=None):
    method_cls = resolve_method(method)
    return method_cls(lhs, rhs, degree=degree, source=source)


class MultiStage(Eq):
    """
    Abstract base class for multi-stage time integration methods
    (e.g., Runge-Kutta schemes) in Devito.

    This class represents a symbolic equation of the form `target = rhs`
    and provides a mechanism to associate it with a time integration
    scheme. The specific integration behavior must be implemented by
    subclasses via the `_evaluate` method.

    Parameters
    ----------
    lhs : expr-like
       The left-hand side of the equation, typically a time-updated Function
       (e.g., `u.forward`).
    rhs : expr-like, optional
       The right-hand side of the equation to integrate. Defaults to 0.
    subdomain : SubDomain, optional
       A subdomain over which the equation applies.
    coefficients : dict, optional
       Optional dictionary of symbolic coefficients for the integration.
    implicit_dims : tuple, optional
       Additional dimensions that should be treated implicitly in the equation.
    **kwargs : dict
       Additional keyword arguments, such as time integration method selection.

    Notes
    -----
    Subclasses must override the `_evaluate()` method to return a sequence
    of update expressions for each stage in the integration process.
    """

    def __new__(cls, lhs, rhs, degree=None, source=None, **kwargs):
        # Normalize to lists first lhs and rhs
        if not isinstance(lhs, (list, tuple)):
            lhs = [lhs]
        if not isinstance(rhs, (list, tuple)):
            rhs = [rhs]

        # Convert to tuples for immutability
        lhs_tuple = tuple([i.function for i in lhs])
        rhs_tuple = tuple(rhs)

        obj = super().__new__(cls, lhs_tuple[0], rhs_tuple[0], **kwargs)

        # Store all equations as immutable tuples
        obj._eq = tuple(Eq(lhs, rhs) for lhs, rhs in zip(lhs_tuple, rhs_tuple))
        obj._lhs = lhs_tuple
        obj._rhs = rhs_tuple
        obj._deg = degree
        # Convert source to tuple of tuples for immutability
        obj._src = tuple(tuple(item)
                         for item in source) if source is not None else None
        obj._t = lhs_tuple[0].grid.time_dim
        obj._dt = obj._t.spacing
        obj._n_eq = len(lhs_tuple)

        return obj

    @property
    def eq(self):
        """Return the full tuple of equations."""
        return self._eq

    @property
    def lhs(self):
        """Return tuple of left-hand sides."""
        return self._lhs

    @property
    def rhs(self):
        """Return tuple of right-hand sides."""
        return self._rhs

    @property
    def deg(self):
        """Return the degree parameter."""
        return self._deg

    @property
    def src(self):
        """Return the source parameter as tuple of tuples (immutable)."""
        return self._src

    @property
    def t(self):
        """Return the time (t) parameter."""
        return self._t

    @property
    def dt(self):
        """Return the time step (dt) parameter."""
        return self._dt

    @property
    def n_eq(self):
        """Return the number of equations."""
        return self._n_eq

    def _evaluate(self, **kwargs):
        raise NotImplementedError(
            f"_evaluate() must be implemented in the subclass {self.__class__.__name__}")


class RungeKutta(MultiStage):
    """
    Base class for explicit Runge-Kutta (RK) time integration methods defined
    via a Butcher tableau.

    This class handles the general structure of RK schemes by using
    the Butcher coefficients (`a`, `b`, `c`) to expand a single equation into
    a series of intermediate stages followed by a final update. Subclasses
    must define `a`, `b`, and `c` as class attributes.

    Parameters
    ----------
    a : tuple of tuple of float
        The coefficient matrix representing stage dependencies.
    b : tuple of float
        The weights for the final combination step.
    c : tuple of float
        The time shifts for each intermediate stage (often the row sums of `a`).

    Attributes
    ----------
    a : tuple[tuple[float, ...], ...]
        Butcher tableau `a` coefficients (stage coupling).
    b : tuple[float, ...]
        Butcher tableau `b` coefficients (weights for combining stages).
    c : tuple[float, ...]
        Butcher tableau `c` coefficients (stage time positions).
    s : int
        Number of stages in the RK method, inferred from `b`.
    """

    CoeffsBC = tuple[float | np.number, ...]
    CoeffsA = tuple[CoeffsBC, ...]

    def __init__(self, a: CoeffsA, b: CoeffsBC, c: CoeffsBC, lhs, rhs, **kwargs) -> None:
        self.a, self.b, self.c = a, b, c

    @property
    def s(self):
        return len(self.b)

    def _evaluate(self, **kwargs):
        """
        Generate the stage-wise equations for a Runge-Kutta time integration method.

        This method takes a single equation of the form `Eq(u.forward, rhs)` and
        expands it into a sequence of intermediate stage evaluations and a final
        update equation according to the Runge-Kutta coefficients `a`, `b`, and `c`.

        Returns
        -------
        list of Devito Eq objects
            A list of SymPy Eq objects representing:
            - `s` stage equations of the form `k_i = rhs evaluated at intermediate state`
            - 1 final update equation of the form `u.forward = u + dt * sum(b_i * k_i)`
        """

        sregistry = kwargs.get('sregistry')
        # Create temporary Arrays to hold each stage
        k = [[Array(name=f'{sregistry.make_name(prefix='k')}', dimensions=self.lhs[j].grid.dimensions, grid=self.lhs[j].grid, dtype=self.lhs[j].dtype) for i in range(self.s)]
             for j in range(self.n_eq)]

        stage_eqs = []

        # Build each stage
        for i in range(self.s):
            u_temp = [self.lhs[l] + self.dt * sum(aij * kj for aij, kj in zip(
                self.a[i][:i], k[l][:i])) for l in range(self.n_eq)]
            t_shift = self.t + self.c[i]

            # Evaluate RHS at intermediate value
            stage_rhs = [uxreplace(self.rhs[l], {**{self.lhs[m]: u_temp[m] for m in range(
                self.n_eq)}, self.t: t_shift}) for l in range(self.n_eq)]
            stage_eqs.extend([Eq(k[l][i], stage_rhs[l])
                             for l in range(self.n_eq)])

        # Final update: u.forward = u + dt * sum(b_i * k_i)
        u_next = [self.lhs[l] + self.dt *
                  sum(bi * ki for bi, ki in zip(self.b, k[l])) for l in range(self.n_eq)]
        stage_eqs.extend([Eq(self.lhs[l].forward, u_next[l])
                         for l in range(self.n_eq)])

        return stage_eqs


@register_method(aliases=['RK44'])
class RungeKutta44(RungeKutta):
    """
    Classic 4th-order Runge-Kutta (RK4) time integration method.

    This class implements the classic explicit Runge-Kutta method of order 4 (RK44).

    Attributes
    ----------
    a : tuple[tuple[float, ...], ...]
        Coefficients of the `a` matrix for intermediate stage coupling.
    b : tuple[float, ...]
        Weights for final combination.
    c : tuple[float, ...]
        Time positions of intermediate stages.
    """
    a = ((0, 0, 0, 0),
         (1/2, 0, 0, 0),
         (0, 1/2, 0, 0),
         (0, 0, 1, 0))
    b = (1/6, 1/3, 1/3, 1/6)
    c = (0, 1/2, 1/2, 1)

    def __init__(self, lhs, rhs, **kwargs):
        super().__init__(a=self.a, b=self.b, c=self.c, lhs=lhs, rhs=rhs, **kwargs)


@register_method(aliases=['RK32'])
class RungeKutta32(RungeKutta):
    """
    3 stages 2nd-order Runge-Kutta (RK32) time integration method.

    This class implements the 3-stages explicit Runge-Kutta method of order 2 (RK32).

    Attributes
    ----------
    a : list[list[float]]
        Coefficients of the `a` matrix for intermediate stage coupling.
    b : list[float]
        Weights for final combination.
    c : list[float]
        Time positions of intermediate stages.
    """
    a = ((0, 0, 0),
         (1/2, 0, 0),
         (0, 1/2, 0))
    b = (0, 0, 1)
    c = (0, 1/2, 1/2)

    def __init__(self, lhs, rhs, **kwargs):
        super().__init__(a=self.a, b=self.b, c=self.c, lhs=lhs, rhs=rhs, **kwargs)


@register_method(aliases=['RK97'])
class RungeKutta97(RungeKutta):
    """
    9 stages 7th-order Runge-Kutta (RK97) time integration method.

    This class implements the 9-stages explicit Runge-Kutta method of order 7 (RK97).

    Attributes
    ----------
    a : list[list[float]]
        Coefficients of the `a` matrix for intermediate stage coupling.
    b : list[float]
        Weights for final combination.
    c : list[float]
        Time positions of intermediate stages.
    """
    a = ((0, 0, 0, 0, 0, 0, 0, 0, 0),
         (4/63, 0, 0, 0, 0, 0, 0, 0, 0),
         (1/42, 1/14, 0, 0, 0, 0, 0, 0, 0),
         (1/28, 0, 3/28, 0, 0, 0, 0, 0, 0),
         (12551/19652, 0, -48363/19652, 10976/4913, 0, 0, 0, 0, 0),
         (-36616931/27869184, 0, 2370277/442368, -255519173 /
          63700992, 226798819/445906944, 0, 0, 0, 0),
         (-10401401/7164612, 0, 47383/8748, -4914455 /
          1318761, -1498465/7302393, 2785280/3739203, 0, 0, 0),
         (181002080831/17500000000, 0, -14827049601/400000000, 23296401527134463/857600000000000,
          2937811552328081/949760000000000, -243874470411/69355468750, 2857867601589/3200000000000),
         (-228380759/19257212, 0, 4828803/113948, -331062132205/10932626912, -12727101935/3720174304,
          22627205314560/4940625496417, -268403949/461033608, 3600000000000/19176750553961))
    b = (95/2366, 0, 0, 3822231133/16579123200, 555164087/2298419200, 1279328256/9538891505,
         5963949/25894400, 50000000000/599799373173, 28487/712800)
    c = (0, 4/63, 2/21, 1/7, 7/17, 13/24, 7/9, 91/100, 1)

    def __init__(self, lhs, rhs, **kwargs):
        super().__init__(a=self.a, b=self.b, c=self.c, lhs=lhs, rhs=rhs, **kwargs)


@register_method(aliases=['HORK_EXP'])
class HighOrderRungeKuttaExponential(MultiStage):
    # In construction
    """
    n stages Runge-Kutta (HORK) time integration method.

    This class implements the arbitrary high-order explicit Runge-Kutta method.

    Attributes
    ----------
    a : list[list[float]]
        Coefficients of the `a` matrix for intermediate stage coupling.
    b : list[float]
        Weights for final combination.
    c : list[float]
        Time positions of intermediate stages.
    """

    def source_derivatives(self, src_index, **kwargs):

        # Compute the base wavelet function
        f_deriv = [[src[1] for src in self.src]]

        # Compute derivatives up to order p
        for _ in range(self.deg - 1):
            f_deriv.append([deriv.diff(self.t) for deriv in f_deriv[-1]])

        f_deriv.reverse()
        return f_deriv

    def ssprk_alpha(self, mu=1):
        """
        Computes the coefficients for the Strong Stability Preserving Runge-Kutta (SSPRK) method.

        Parameters:
        mu : float
            Theoretically, it should be the inverse of the CFL condition (typically mu=1 for best performance).
            In practice, mu=1 works better.
        degree : int
            Degree of the polynomial used in the time-stepping scheme.

        Returns:
        numpy.ndarray
            Array of SSPRK coefficients.
        """

        alpha = [0] * self.deg
        alpha[0] = 1.0  # Initial coefficient

        # recurrence relation to compute the HORK coefficients following the formula in Gottlieb and Gottlieb (2002)
        for i in range(1, self.deg):
            alpha[i] = 1 / (mu * (i + 1)) * alpha[i - 1]
            alpha[1:i] = [1 / (mu * j) * alpha[j - 1] for j in range(1, i)]
            alpha[0] = 1 - sum(alpha[1:i + 1])

        return alpha

    def source_inclusion(self, current_state, stage_values, e_p, **integration_params):
        """
        Include source terms in the time integration step.

        This method applies source term contributions to the right-hand side
        of the differential equations during time integration, accounting for
        time derivatives of the source function and expansion coefficients.

        Parameters
        ----------
        current_state : list
            Current state variables (u).
        stage_values : list
            Current stage values (k).
        e_p : list
            Expansion coefficients for stability control.
        **integration_params : dict
            Integration parameters containing 't', 'dt', 'mu', 'src_index',
            'src_deriv', 'n_eq'.

        Returns
        -------
        tuple
            (modified_rhs, updated_e_p) - Updated right-hand side
            equations and modified expansion coefficients.
        """
        # Extract integration parameters
        mu = integration_params['mu']
        src_index = integration_params['src_index']
        src_deriv = integration_params['src_deriv']
        n_eq = integration_params['n_eq']

        # Build base right-hand side by substituting current stage values
        src_lhs = [uxreplace(self.rhs[i], {current_state[m]: stage_values[m] for m in range(n_eq)})
                   for i in range(n_eq)]

        # Apply source term contributions if sources exist
        if self.src is not None:
            p = len(src_deriv)

            # Add source contributions for each derivative order
            for i in range(p):
                if e_p[i] != 0:
                    for j, idx in enumerate(src_index):
                        # Add weighted source derivative contribution
                        source_contribution = (self.src[j][0]
                                               * src_deriv[i][j].subs({self.t: self.t * self.dt})
                                               * e_p[i])
                        src_lhs[idx] += source_contribution

            # Update expansion coefficients for next stage
            e_p = [e_p[i] + mu*self.dt*e_p[i + 1]
                   for i in range(p - 1)] + [e_p[-1]]

        return src_lhs, e_p

    def _evaluate(self, **kwargs):
        """
        Generate the stage-wise equations for a Runge-Kutta time integration method.

        This method takes a single equation of the form `Eq(u.forward, rhs)` and
        expands it into a sequence of intermediate stage evaluations and a final
        update equation according to the Runge-Kutta coefficients `a`, `b`, and `c`.

        Returns
        -------
        list of Eq
            A list of SymPy Eq objects representing:
            - `s` stage equations of the form `k_i = rhs evaluated at intermediate state`
            - 1 final update equation of the form `u.forward = u + dt * sum(b_i * k_i)`
        """

        sregistry = kwargs.get('sregistry')
        # Create a temporary Array for each variable to save the time stages
        # k = [Array(name=f'{sregistry.make_name(prefix='k')}', dimensions=u[i].grid.dimensions, grid=u[i].grid, dtype=u[i].dtype) for i in range(n_eq)]
        k = [Function(name=f'{sregistry.make_name(prefix='k')}', grid=self.lhs[i].grid,
                      space_order=2, time_order=1, dtype=self.lhs[i].dtype) for i in range(self.n_eq)]
        k_old = [Function(name=f'{sregistry.make_name(prefix='k')}', grid=self.lhs[i].grid,
                          space_order=2, time_order=1, dtype=self.lhs[i].dtype) for i in range(self.n_eq)]

        # Compute SSPRK coefficients
        mu = 1
        alpha = self.ssprk_alpha(mu=mu)

        # Initialize symbolic differentiation for source terms
        field_map = {val: i for i, val in enumerate(self.lhs)}
        if self.src is not None:
            src_index = [field_map[src[2]] for src in self.src]
            src_deriv = self.source_derivatives(src_index, **kwargs)
        else:
            src_index = None
            src_deriv = None

        # Expansion coefficients for stability control
        e_p = [0] * self.deg
        eta = 1
        e_p[-1] = 1 / eta

        stage_eqs = [Eq(ki, ui) for ki, ui in zip(k, self.lhs)]
        stage_eqs.extend([Eq(lhs_i.forward, lhs_i*alpha[0]) for lhs_i in self.lhs])

        # Prepare integration parameters for source inclusion
        integration_params = {'mu': mu, 'src_index': src_index,
                              'src_deriv': src_deriv, 'n_eq': self.n_eq}

        # Build each stage
        for i in range(1, self.deg - 1):
            stage_eqs.extend([Eq(k_old_j, k_j) for k_old_j, k_j in zip(k_old, k)])
            src_lhs, e_p = self.source_inclusion(self.lhs, k_old, e_p, **integration_params)
            stage_eqs.extend([Eq(k_j, k_old_j+mu*self.dt*src_lhs_j) 
                              for k_j, k_old_j, src_lhs_j in zip(k, k_old, src_lhs)])
            stage_eqs.extend([Eq(lhs_j.forward, lhs_j.forward+k_j*alpha[i]) 
                              for lhs_j, k_j in zip(self.lhs, k)])

        # Final Runge-Kutta updates
        stage_eqs.extend([Eq(k_old_j, k_j) for k_old_j, k_j in zip(k_old, k)])
        src_lhs, e_p = self.source_inclusion(self.lhs, k_old, e_p, **integration_params)
        stage_eqs.extend([Eq(k_j, k_old_j+mu*self.dt*src_lhs_j)
                         for k_j, k_old_j, src_lhs_j in zip(k, k_old, src_lhs)])

        stage_eqs.extend([Eq(k_old_j, k_j) for k_old_j, k_j in zip(k_old, k)])
        src_lhs, _ = self.source_inclusion(self.lhs, k_old, e_p, **integration_params)
        stage_eqs.extend([Eq(k_j, k_old_j+mu*self.dt*src_lhs_j)
                         for k_j, k_old_j, src_lhs_j in zip(k, k_old, src_lhs)])

        # Compute final approximation
        stage_eqs.extend([Eq(lhs_j.forward, lhs_j.forward+k_j*alpha[self.deg-1])
                         for lhs_j, k_j in zip(self.lhs, k)])

        return stage_eqs

method_registry = MappingProxyType(method_registry)