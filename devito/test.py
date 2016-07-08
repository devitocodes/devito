import numpy as np
from sympy import IndexedBase
from tools import aligned
from random import randint


f = "/home/pp3613/mdevito/devito/disk/" + str(randint(0,100000))
print f
pointer = aligned(np.memmap(filename=f, dtype=np.float64, mode='w+', shape=(1000,1000,1000), order='C'), alignment=64)
print pointer[3][4][99]
