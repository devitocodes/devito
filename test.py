'''
Created on 24 Mar 2016

@author: navjotkukreja
'''

from generator import Generator
import numpy as np

g = Generator(np.ones((2,3,4), dtype=np.float64))
g.execute("basic.cpp")