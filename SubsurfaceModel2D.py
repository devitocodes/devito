#! /bin/usr/env python
import numpy as np


class SubsurfaceModel2D:
    ''' Subsurface model object. Used to read, write and manage model data.
    '''
    def __init__(self, filename=None, vp=None, nshots=None, nrecv=None, dtype=None, shape=None, offset=None):
        self.filename = filename
        self.vp = vp
        self.nshots = nshots
        self.nrecv = nrecv
        self.dtype = dtype
        self.shape = shape
        self.offset = offset
        if filename is not None:
            self.field = self.read(filename, nshots, nrecv, dtype, shape, offset)

    def create_model(self, origin, spacing, vp, cfl=0.45):
        self.spacing = spacing
        self.origin = origin
        self.vp = vp
        self.dt_num = cfl * min(self.spacing) / np.max(vp)

    def get_origin(self):
        return self.origin

    def get_spacing(self):
        return self.spacing

    def get_dimensions(self):
        return self.vp.shape

    def get_vp(self):
        return 1./(self.vp**2)

    def set_vp(self, vp):
        self.vp = vp
    
    def get_shape(self):
        return self.vp.shape

    def get_critical_dt(self):
        return self.dt_num

    def read(self, filename, nshots, nrecv, dtype, shape, offset):
        ''' Readers for raw binary, seg-y'''
        try:
            reader = BinaryReader(filename, nshots, nrecv, dtype, shape, offset)
            field = np.sqrt(1./reader.read())
        except:
            raise NotImplementedError("Test missing")
        return field

if __name__ == "__main__":
    raise NotImplementedError("Test missing")