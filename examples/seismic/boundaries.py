from __future__ import division

from devito import Dimension, x, y, z, t, Forward
from devito.exceptions import InvalidArgument

from sympy import Eq

__all__ = ['ABC']


# assuming (t/time, x, y, z), for now, generalization gonna be tricky

class ABC(object):
    """
    Absrobing boundary layer for second-order scalar acoustic wave equations
    :param model : model structure containing the boundary layer size and velocity model
    :param field : propagated field as a TimdeData object
    :param m : square slowness as a DenseDat object
    :param taxis : Forward or Backward, defines the propagation axis
    """

    def __init__(self, model, field, m, taxis=Forward, **kwargs):
        self.nbpml = model.nbpml
        self.full_shape = model.shape_domain
        self.p_abc = Dimension(name="abc", size=self.nbpml)
        self.ndim = model.dim
        self.field = field
        self.m = m
        self.taxis = taxis
        self.fs = Dimension(name="fs", size=model.nbpml)
        self.freesurface = kwargs.get("freesurface", False)

    @property
    def abc_eq(self):
        """
        Equation of the absorbing boundary condition as a complement of the PDE
        :param val: symbolic value of the dampening profile
        :return: Symbolic equation inside the boundary layer
        """
        s = t.spacing
        next = self.field.forward if self.taxis is Forward else self.field.backward
        prev = self.field.backward if self.taxis is Forward else self.field.forward
        return Eq(next, self.m / (self.m + s * self.damp_profile) * next +
                  s * self.damp_profile / (self.m + s * self.damp_profile) * prev)

    @property
    def free_surface(self):
        """
        Free surface expression. Mirrors the negative wavefield above the sea level
        :return: SYmbolic equation of the free surface
        """
        ind_fs = self.field.indices[-1]
        next = self.field.forward if self.taxis is Forward else self.field.backward
        return [Eq(next.subs({ind_fs: self.fs}),
                   - next.subs({ind_fs: 2*self.nbpml - self.fs}))]

    @property
    def abc(self):
        """
        Complete set of expressions for the ABC layers
        :return:
        """
        if self.ndim == 2:
            return self.damp_2d()
        elif self.ndim == 3:
            return self.damp_3d()
        else:
            raise InvalidArgument("Unsupported model shape")

    @property
    def damp_profile(self):
        """
        Dampening profile along a single direction
        :return:
        """
        pos = (self.nbpml - self.p_abc)
        # 1 - exp(-(0.004*pos)**2) first order Taylor (very close to 0)
        # second order would be 1/250*pos)**2 *(1 + 1/2*(1/250*pos)*(1/250*pos))
        profile = (1 / 250 * pos) * (1 / 250 * pos)
        return profile

    def damp_x(self):
        """
        Dampening profile along x
        :return:
        """
        return [self.abc_eq.subs({x: self.p_abc}),
                self.abc_eq.subs({x: self.full_shape[0] - 1 - self.p_abc})]

    def damp_y(self):
        """
        Dampening profile along y
        :return:
        """
        return [self.abc_eq.subs({y: self.p_abc}),
                self.abc_eq.subs({y: self.full_shape[0] - 1 - self.p_abc})]

    def damp_z(self):
        """
        Dampening profile along z
        This seems to need a different Dimension to avoid beeing
        stuck in an infinite loop inside /devito/tools.py", line 123 while loop on queue
        :return:
        """
        p_abcz = Dimension(name="abcz", size=self.nbpml)
        pos = (self.nbpml - p_abcz + 1)
        profile = (1 / 250 * pos) ** 2
        s = t.spacing
        next = self.field.forward if self.taxis is Forward else self.field.backward
        prev = self.field.backward if self.taxis is Forward else self.field.forward
        abc_eq = Eq(next, self.m / (self.m + s * profile) * next +
                    s * profile / (self.m + s * profile) * prev)
        return [abc_eq.subs({z: p_abcz}),
                abc_eq.subs({z: self.full_shape[0] - 1 - p_abcz})]
        # return [self.abc_eq.subs({z: self.p_abc, h: z.spacing}),
        #         self.abc_eq.subs({z: self.full_shape[0] - 1 - self.p_abc,
        #                                h: z.spacing})]

    def damp_2d(self):
        """
        Dampening profiles in 2D w/ w/o freesurface
        :return:
        """
        return self.damp_x() + (self.free_surface if self.freesurface else self.damp_y())

    def damp_3d(self):
        """
        Dampening profiles in 2D w/ w/o freesurface
        :return:
        """
        return self.damp_x() + self.damp_y() +\
            (self.free_surface if self.freesurface else self.damp_z())
