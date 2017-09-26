from devito import Dimension, x, y, z, t, Forward, Backward
from devito.exceptions import InvalidArgument

import numpy as np
from sympy import Eq, sin, sqrt
from sympy.abc import h

__all__ = ['ABC']


# assuming (t/time, x, y, z), for now, generalization gonna be tricky

class ABC(object):
    """
    Absrobing boundary layer for second-order scalar acoustic wave equations
    :param model : model structure containing the boundary layer size and velocity model
    :param field : propagated field as a TimdeData object
    :param m : square slowness
    :param taxis : Forward or Backward, defines the propagation axis
    """

    def __init__(self, model, field, m, taxis=Forward, **kwargs):
        self.nbpml = model.nbpml
        self.full_shape = model.shape_domain
        self.p_abc = Dimension(name="abc", size=self.nbpml)
        self.ndim = len(model.shape)
        self.dampcoeff = np.log(1.0 / 0.001) / (40. * h)
        self.field = field
        self.m = m
        pos = abs((self.nbpml - self.p_abc + 1) / float(self.nbpml))
        self.val = (sqrt(1 / m) * self.dampcoeff *
                    (pos - sin(2 * np.pi * pos) / (2 * np.pi)))
        self.taxis = taxis
        self.fs = Dimension(name="fs", size=model.nbpml)
        self.freesurface = kwargs.get("freesurface", False)

    @property
    def abc_eq(self):
        s = t.spacing
        next = self.field.forward if self.taxis is Forward else self.field.backward
        prev = self.field.backward if self.taxis is Forward else self.field.forward
        return Eq(next, self.m / (self.m + s * self.val) * next +
                  s * self.val / (self.m + s * self.val) * prev)

    @property
    def free_surface(self):
        ind_fs = self.field.indices[-1]
        next = self.field.forward if self.taxis is Forward else self.field.backward
        return [Eq(next.subs({ind_fs: self.fs}),
                   - next.subs({ind_fs: self.nbpml - self.fs}))]

    @property
    def abc(self):
        if len(self.full_shape) == 2:
            return self.damp_2d()
        elif len(self.full_shape) == 3:
            return self.damp_3d()
        else:
            raise InvalidArgument("Unsupported model shape")

    def damp_x(self):
        return [self.abc_eq.subs({x: self.p_abc, h: x.spacing}),
                self.abc_eq.subs({x: self.full_shape[0] - 1 - self.p_abc,
                                  h: x.spacing})]

    def damp_y(self):
        return [self.abc_eq.subs({y: self.p_abc, h: y.spacing}),
                self.abc_eq.subs({y: self.full_shape[0] - 1 - self.p_abc,
                                  h: y.spacing})]

    def damp_z(self):
        return [self.abc_eq.subs({z: self.p_abc, h: z.spacing}),
                self.abc_eq.subs({z: self.full_shape[0] - 1 - self.p_abc,
                                  h: z.spacing})]

    def damp_2d(self):
        return self.damp_x() + (self.free_surface if self.freesurface else self.damp_y())

    def damp_3d(self):
        return self.damp_x() + self.damp_y() +\
               (self.free_surface if self.freesurface else self.damp_z())
