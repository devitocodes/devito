#! /bin/usr/env python
import os
import numpy as np


class SeismicData:
    ''' Generic container for Seismic Data
    '''
    def __init__(self, files=None, dtype = None):
        # Reimplement to perform a broadcast.
        #self.shots = opesci_segy.create_index(files)
        self.receiver_coords = np.ndarray((1,3), dtype = dtype)

    def read(self):
        return self

    def set_source(self, time_series, dt, location):
        self.source = time_series
        self.source_coords = location
    
    def get_source(self, t = None):
        if t is None:
            return self.source
        return self.source[t]
      
    def reinterpolate(self, x):
        pass