from .equation import Eq
from .dense import Function
from devito.symbolics import uxreplace

# from devito.ir.support import SymbolRegistry

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

    def __init__(self, a=None, b=None, c=None, **kwargs):
        self.a, self.b, self.c = self._validate(a, b, c)

    def _validate(self, a, b, c):
        if a is None or b is None or c is None:
            raise ValueError("RK subclass must define class attributes of the Butcher's array a, b, and c")
        return a, b, c

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
    It uses four intermediate stages and specific Butcher coefficients to achieve
    high accuracy while remaining explicit.

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
        super().__init__(a=self.a, b=self.b, c=self.c)

