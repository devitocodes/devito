from devito import Function, Eq
from devito.symbolics import uxreplace
from sympy import Basic


class MultiStage(Basic):
    def __new__(cls, eq, method):
        assert isinstance(eq, Eq)
        return Basic.__new__(cls, eq, method)

    @property
    def eq(self):
        return self.args[0]

    @property
    def method(self):
        return self.args[1]


class RK(Basic):
    """
    A class representing an explicit Runge-Kutta method via its Butcher tableau.

    Parameters
    ----------
    a : list[list[float]]
        Lower-triangular coefficient matrix (stage dependencies).
    b : list[float]
        Weights for the final combination step.
    c : list[float]
        Weights for the stages time step.
    """

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.s = len(b)   # number of stages

        self._validate()

    def _validate(self):
        assert len(self.a) == self.s, "'a' must have s rows"
        for i, row in enumerate(self.a):
            assert len(row) == i, f"Row {i} in 'a' must have {i} entries for explicit RK"

    def expand_stages(self, base_eq, eq_num=0):
        """
        Expand a single Eq into a list of stage-wise Eqs for this RK method.

        Parameters
        ----------
        base_eq : Eq
            The equation Eq(u.forward, rhs) to be expanded into RK stages.
        eq_number : integer, optional
            The equation number to idetify the k_i's stages

        Returns
        -------
        list of Eq
            Stage-wise equations: [k0=..., k1=..., ..., u.forward=...]
        """
        u = base_eq.lhs.function
        rhs = base_eq.rhs
        grid = u.grid
        dt = grid.stepping_dim.spacing
        t = grid.time_dim

        # Create temporary Functions to hold each stage
        k = [Function(name=f'k{eq_num}{i}', grid=grid, space_order=u.space_order, dtype=u.dtype)
             for i in range(self.s)]

        stage_eqs = []

        # Build each stage
        for i in range(self.s):
            u_temp = u
            for j in range(i):
                if self.a[i][j] != 0:
                    u_temp += self.a[i][j] * dt * k[j]
            t_shift = t + self.c[i] * dt

            # Evaluate RHS at intermediate value
            stage_rhs = uxreplace(rhs, {u: u_temp, t: t_shift})
            stage_eqs.append(Eq(k[i], stage_rhs))

        # Final update: u.forward = u + dt * sum(bᵢ * kᵢ)
        u_next = u
        for i in range(self.s):
            u_next += self.b[i] * dt * k[i]
        stage_eqs.append(Eq(u.forward, u_next))

        return stage_eqs

    # ---- Named methods for convenience ----
    @classmethod
    def RK44(cls):
        """Classical Runge-Kutta of 4 stages and 4th order"""
        a = [
            [],
            [1 / 2],
            [0, 1 / 2],
            [0, 0, 1]
        ]
        b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        c = [0, 1 / 2, 1 / 2, 1]
        return cls(a, b, c)


