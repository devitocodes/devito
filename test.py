from AcousticWave2D_codegen import AcousticWave2D
import numpy as np


class DummyModel(object):
    def get_critical_dt(self):
        return 0.001

    def get_spacing(self):
        return (0.1, 0.1)

    def get_origin(self):
        return (0, 0)

    def get_dimensions(self):
        return (20, 20)

    def get_vp(self):
        return np.zeros(self.get_dimensions())


class DummyData(object):
    xrec = 5

    def __init__(self):
        t0 = 1/.010
        f0 = .010
        self.src = []
        for t in range(50):
            r = (np.pi*f0*(t-t0))
            self.src.append((1-2.*r**2)*np.exp(-r**2))

    def get_source_loc(self):
        return (0, 0)

    def get_source(self):
        return self.src

a = AcousticWave2D(DummyModel(), DummyData())
a.prepare(10)
print a.Forward()
