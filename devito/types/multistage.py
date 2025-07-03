from .equation import Eq
from .dense import Function
from devito.symbolics import uxreplace
from numpy import number

from .array import Array  # Trying Array


method_registry = {}

def register_method(cls):
    method_registry[cls.__name__] = cls
    return cls


def resolve_method(method):
    try:
        return method_registry[method]
    except KeyError:
        raise ValueError(f"The time integrator '{method}' is not implemented.")


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

    def __new__(cls, lhs, rhs=0, subdomain=None, coefficients=None, implicit_dims=None, **kwargs):
        return super().__new__(cls, lhs, rhs=rhs, subdomain=subdomain, coefficients=coefficients, implicit_dims=implicit_dims, **kwargs)

    def _evaluate(self, **kwargs):
        raise NotImplementedError(
            f"_evaluate() must be implemented in the subclass {self.__class__.__name__}")


class RK(MultiStage):
    """
    Base class for explicit Runge-Kutta (RK) time integration methods defined
    via a Butcher tableau.

    This class handles the general structure of RK schemes by using
    the Butcher coefficients (`a`, `b`, `c`) to expand a single equation into
    a series of intermediate stages followed by a final update. Subclasses
    must define `a`, `b`, and `c` as class attributes.

    Parameters
    ----------
    a : list of list of float
        The coefficient matrix representing stage dependencies.
    b : list of float
        The weights for the final combination step.
    c : list of float
        The time shifts for each intermediate stage (often the row sums of `a`).

    Attributes
    ----------
    a : list[list[float]]
        Butcher tableau `a` coefficients (stage coupling).
    b : list[float]
        Butcher tableau `b` coefficients (weights for combining stages).
    c : list[float]
        Butcher tableau `c` coefficients (stage time positions).
    s : int
        Number of stages in the RK method, inferred from `b`.
    """

    def __init__(self, a: list[list[float | number]], b: list[float | number], c: list[float | number], **kwargs) -> None:
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
        list of Eq
            A list of SymPy Eq objects representing:
            - `s` stage equations of the form `k_i = rhs evaluated at intermediate state`
            - 1 final update equation of the form `u.forward = u + dt * sum(b_i * k_i)`
        """

        u = self.lhs.function
        rhs = self.rhs
        grid = u.grid
        t = grid.time_dim
        dt = t.spacing

        # Create temporary Functions to hold each stage
        # k = [Array(name=f'{kwargs.get('sregistry').make_name(prefix='k')}', dimensions=grid.shape, grid=grid, dtype=u.dtype) for i in range(self.s)]  # Trying Array
        k = [Function(name=f'{kwargs.get('sregistry').make_name(prefix='k')}', grid=grid, space_order=u.space_order, dtype=u.dtype)
             for i in range(self.s)]

        stage_eqs = []

        # Build each stage
        for i in range(self.s):
            u_temp = u + dt * sum(aij * kj for aij, kj in zip(self.a[i][:i], k[:i]))
            t_shift = t + self.c[i] * dt

            # Evaluate RHS at intermediate value
            stage_rhs = uxreplace(rhs, {u: u_temp, t: t_shift})
            stage_eqs.append(Eq(k[i], stage_rhs))

        # Final update: u.forward = u + dt * sum(b_i * k_i)
        u_next = u + dt * sum(bi * ki for bi, ki in zip(self.b, k))
        stage_eqs.append(Eq(u.forward, u_next))

        return stage_eqs


@register_method
class RK44(RK):
    """
    Classic 4th-order Runge-Kutta (RK4) time integration method.

    This class implements the classic explicit Runge-Kutta method of order 4 (RK44).

    Attributes
    ----------
    a : list[list[float]]
        Coefficients of the `a` matrix for intermediate stage coupling.
    b : list[float]
        Weights for final combination.
    c : list[float]
        Time positions of intermediate stages.
    """
    a = [[0, 0, 0, 0],
         [1/2, 0, 0, 0],
         [0, 1/2, 0, 0],
         [0, 0, 1, 0]]
    b = [1/6, 1/3, 1/3, 1/6]
    c = [0, 1/2, 1/2, 1]

    def __init__(self, *args, **kwargs):
        super().__init__(a=self.a, b=self.b, c=self.c, **kwargs)


@register_method
class RK32(RK):
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
    a = [[0, 0, 0],
         [1/2, 0, 0],
         [0, 1/2, 0]]
    b = [0, 0, 1]
    c = [0, 1/2, 1/2]

    def __init__(self, *args, **kwargs):
        super().__init__(a=self.a, b=self.b, c=self.c, **kwargs)


@register_method
class RK97(RK):
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
    a = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [4/63, 0, 0, 0, 0, 0, 0, 0, 0],
         [1/42, 1/14, 0, 0, 0, 0, 0, 0, 0],
         [1/28, 0, 3/28, 0, 0, 0, 0, 0, 0],
         [12551/19652, 0, -48363/19652, 10976/4913, 0, 0, 0, 0, 0],
         [-36616931/27869184, 0, 2370277/442368, -255519173/63700992, 226798819/445906944, 0, 0, 0, 0],
         [-10401401/7164612, 0, 47383/8748, -4914455/1318761, -1498465/7302393, 2785280/3739203, 0, 0, 0],
         [181002080831/17500000000, 0, -14827049601/400000000, 23296401527134463/857600000000000,
          2937811552328081/949760000000000, -243874470411/69355468750, 2857867601589/3200000000000],
         [-228380759/19257212, 0, 4828803/113948, -331062132205/10932626912, -12727101935/3720174304,
          22627205314560/4940625496417, -268403949/461033608, 3600000000000/19176750553961]]
    b = [95/2366, 0, 0, 3822231133/16579123200, 555164087/2298419200, 1279328256/9538891505,
         5963949/25894400, 50000000000/599799373173, 28487/712800]
    c = [0, 4/63, 2/21, 1/7, 7/17, 13/24, 7/9, 91/100, 1]

    def __init__(self, *args, **kwargs):
        super().__init__(a=self.a, b=self.b, c=self.c, **kwargs)


@register_method
class HORK(MultiStage):
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


    def ssprk_alpha(mu=1, **kwargs):
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
        degree=kwargs.get('degree')

        alpha = [0]*degree
        alpha[0] = 1.0  # Initial coefficient

        for i in range(1, degree):
            alpha[i] = 1 / (mu * (i + 1)) * alpha[i - 1]
            alpha[1:i] = 1 / (mu * list(range(1, i))) * alpha[:i - 1]
            alpha[0] = 1 - sum(alpha[1:i + 1])

        return alpha

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

        u = self.lhs.function
        rhs = self.rhs
        grid = u.grid
        t = grid.time_dim
        dt = t.spacing

        an_eq = range(len(U0))

        # Compute SSPRK coefficients
        alpha = np.array(ssprk_alpha(mu, degree), dtype=np.float64)

        # Initialize symbolic differentiation for source terms
        t_var = sym.Symbol('t_var')
        src_deriv = aux_fun.derivates_f(degree, f0)

        # Expansion coefficients for stability control
        e_p = [0] * degree
        e_p[-1] = 1 / eta

        # Initialize approximation and auxiliary variable
        approx = [U0[i] * alpha[0] for i in n_eq]
        aux = U0

        # Perform Runge-Kutta steps
        for i in range(1, degree - 1):
            system_op, e_p = sys_op_extended(aux, x, y, z, param_fun, system, fd_order, src_spat, src_deriv, t, dt, t_var, e_p)
            aux = [aux[j] + mu * dt * system_op[j] for j in n_eq]
            approx = [approx[j] + aux[j] * alpha[i] for j in n_eq]

        # Final Runge-Kutta updates
        system_op, e_p = sys_op_extended(aux, x, y, z, param_fun, system, fd_order, src_spat, src_deriv, t, dt, t_var, e_p)
        aux = [aux[i] + mu * dt * system_op[i] for i in n_eq]
        system_op, e_p = sys_op_extended(aux, x, y, z, param_fun, system, fd_order, src_spat, src_deriv, t, dt, t_var, e_p)
        aux = [aux[i] + mu * dt * system_op[i] for i in n_eq]

        # Compute final approximation
        approx = [approx[i] + aux[i] * alpha[degree - 1] for i in n_eq]

        # Generate final PDE system
        return [dv.Eq(U0[i].forward, approx[i]) for i in n_eq]

        # Create temporary Functions to hold each stage
        # k = [Array(name=f'{kwargs.get('sregistry').make_name(prefix='k')}', dimensions=grid.shape, grid=grid, dtype=u.dtype) for i in range(self.s)]  # Trying Array
        k = [Function(name=f'{kwargs.get('sregistry').make_name(prefix='k')}', grid=grid, space_order=u.space_order, dtype=u.dtype)
             for i in range(self.s)]

        stage_eqs = []

        # Build each stage
        for i in range(self.s):
            u_temp = u + dt * sum(aij * kj for aij, kj in zip(self.a[i][:i], k[:i]))
            t_shift = t + self.c[i] * dt

            # Evaluate RHS at intermediate value
            stage_rhs = uxreplace(rhs, {u: u_temp, t: t_shift})
            stage_eqs.append(Eq(k[i], stage_rhs))

        # Final update: u.forward = u + dt * sum(b_i * k_i)
        u_next = u + dt * sum(bi * ki for bi, ki in zip(self.b, k))
        stage_eqs.append(Eq(u.forward, u_next))

        return stage_eqs