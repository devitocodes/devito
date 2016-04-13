from generator import Generator
import cgen
from AcousticWave2D_codegen import AcousticWave2D
import numpy as np

class DummyModel(object):
    def get_critical_dt(self):
        return 0.001
    
    def get_spacing(self):
        return (0.1, 0.1)
    
    def get_origin(self):
        return (0,0)
    
    def get_dimensions(self):
        return (20,20)
    
    def get_vp(self):
        return np.zeros(self.get_dimensions())
    
class DummyData(object):
    def get_source_loc(self):
        return (0,0)
    xrec = 5
a = AcousticWave2D(DummyModel(), DummyData())
a.prepare(10)
#a.Forward(10)


