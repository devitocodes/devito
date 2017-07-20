from devito import Dimension, x, y, z, Forward, Backward
from devito.exceptions import InvalidArgument

import numpy as np
from sympy import Eq, sin, sqrt
from sympy.abc import s, h

__all__ = ['ABC']


class ABC(object):

    def __init__(self, model, field, m, taxis=Forward):
        self.nbpml = model.nbpml
        self.full_shape = model.shape_domain
        self.p_abc = Dimension(name="abc", size=self.nbpml)
        self.ndim = len(model.shape)
        self.dampcoeff = np.log(1.0 / 0.001) / (40. * h)
        self.field = field
        self.m = m
        pos = abs((self.nbpml - self.p_abc + 1) / float(self.nbpml))
        self.val = sqrt(1/self.m)*self.dampcoeff * (pos - sin(2 * np.pi * pos) / (2 * np.pi))
        self.taxis = taxis

    @property
    def abc_eq(self):
        if self.taxis == Forward:
            return Eq(self.field.forward, self.m / (self.m + s * self.val) * self.field.forward +
                      s * self.val / (self.m + s * self.val) * self.field.backward)
        elif self.taxis == Backward:
            return Eq(self.field.backward, self.m / (self.m + s * self.val) * self.field.backward +
                      s * self.val / (self.m + s * self.val) * self.field.forward)
        else:
            raise InvalidArgument("Unknown arguments passed: " + ", " + self.taxis)

    def damp_x(self):
        return [self.abc_eq.subs({x: self.p_abc}), self.abc_eq.subs({x: self.full_shape[0] - 1 - self.p_abc})]

    def damp_y(self):
        return [self.abc_eq.subs({y: self.p_abc}), self.abc_eq.subs({y: self.full_shape[0] - 1 - self.p_abc})]

    def damp_z(self):
        return [self.abc_eq.subs({z: self.p_abc}), self.abc_eq.subs({z: self.full_shape[0] - 1 - self.p_abc})]

    def damp_2d(self):
        return self.damp_x() + self.damp_y()

    def damp_3d(self):
        return self.damp_x() + self.damp_y() + self.damp_z()
