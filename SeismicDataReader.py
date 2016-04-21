#! /bin/usr/env python
import os
import numpy as np
import opesci_segy


class SeismicDataReader:
    ''' Reader object that manages the loading of different shots.
    '''
    def __init__(self, files=None):
        # Reimplement to perform a broadcast.
        #self.shots = opesci_segy.create_index(files)
        self.receiver_coords = np.ndarray((1,3))
        self.receiver_coords[0,2] = 10+4

    def read(self):
        return self

    def get_number_of_shots(self):
        return len(self.shots)

    def get_shot(self, shot_id):
        # Returns receivers (traces), receiver coordinates, sample_interval
        return opesci_segy.read_shot(self.shots[shot_id])
    def set_source(self, time_series, dt, location):
        self.source = time_series
        self.source_coords = location
    
    def get_source(self, t = None):
        if t is None:
            return self.source
        return self.source[t]
      
    def reinterpolate(self, x):
        pass