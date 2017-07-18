from devito import Dimension, x, y, z

import numpy as np
from sympy import Eq, sin, solve

__all__ = ['ABC']


class ABC(object):

    def __init__(self, model, field, pde):
        self.nbpml = model.nbpml + field.space_order/2
        self.full_shape = model.shape_domain
        self.p_abc = Dimension(name="abc", size=self.nbpml)
        self.pde = pde
        self.ndim = len(model.shape)
        self.dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40.)
        self.field = field
        pos = abs((self.nbpml - self.p_abc + 1) / float(self.nbpml))
        val = self.dampcoeff * (pos - sin(2 * np.pi * pos) / (2 * np.pi))
        self.eq_abc = Eq(self.field.forward, solve(self.pde + val * self.field.dt, self.field.forward, rational=False)[0])

    def damp_x(self):

        return [self.eq_abc.subs({x: self.p_abc})] + [self.eq_abc.subs({x: self.full_shape[0] - 1 - self.p_abc})]

    def damp_y(self):

        return [self.eq_abc.subs({y: self.p_abc})] + [self.eq_abc.subs({y: self.full_shape[1] - 1 - self.p_abc})]

    def damp_z(self):

        return [self.eq_abc.subs({z: self.p_abc})] + [self.eq_abc.subs({z: self.full_shape[2] - 1 - self.p_abc})]

    def damp_2d(self):
        return self.damp_x() + self.damp_y()

    def damp_3d(self):
        return self.damp_x() + self.damp_y() + self.damp_z()
