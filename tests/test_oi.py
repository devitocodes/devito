import numpy as np
from sympy import Eq, IndexedBase, symbols

from devito.propagator import Propagator


class Test_OI_Calculation(object):

    def test_oi_1(self):
        '''OI = (ADD+MUL)/[(LOAD+STORE)*word_size]; word_size=8(double),4(float)
        Equation = v1[i2][i1] = 3*v2[i2][i1] + 2*v3[i2][i1];
        '''
        load = 3.0
        store = 1.0
        add = 1.0
        mul = 2.0
        dtype = np.float32
        i1, i2 = symbols('i1 i2')
        data = np.arange(50, dtype=np.float32).reshape((10, 5))
        v1 = IndexedBase('v1')
        v2 = IndexedBase('v2')
        v3 = IndexedBase('v3')
        eq = Eq(v1[i2, i1], 3*v2[i2, i1] + 2*v3[i2, i1])

        propagator = Propagator("process", 10, (5,), 0)
        propagator.stencils = (eq,)
        propagator.add_param("v1", data.shape, data.dtype)
        propagator.add_param("v2", data.shape, data.dtype)
        propagator.add_param("v3", data.shape, data.dtype)
        f = propagator.cfunction

        propagator_oi = propagator.get_kernel_oi(dtype)
        hand_oi = (mul+add)/((load+store)*np.dtype(dtype).itemsize)

        arr = np.empty_like(data)
        f(data, data, arr)
        assert(propagator_oi == hand_oi)

    def test_oi_2(self):
        '''OI = (ADD+MUL)/[(LOAD+STORE)*word_size]; word_size=8(double),4(float)
        Equation = v1[i2][i1] = (v1[i2][i1] - 1.1F*v2[i2][i1])/(v3[i2][i1] +
                   7.0e-1F*v4[i2][i1]);
        '''
        load = 4.0
        store = 1.0
        add = 2.0
        mul = 3.0
        dtype = np.float32
        i1, i2 = symbols('i1 i2')
        data = np.arange(50, dtype=np.float32).reshape((10, 5))
        v1 = IndexedBase('v1')
        v2 = IndexedBase('v2')
        v3 = IndexedBase('v3')
        v4 = IndexedBase('v4')
        eq = Eq(v1[i2, i1], (v1[i2, i1] - 1.1*v2[i2, i1])/(0.7*v4[i2, i1] + v3[i2, i1]))

        propagator = Propagator("process", 10, (5,), 0)
        propagator.stencils = (eq, )
        propagator.add_param("v1", data.shape, data.dtype)
        propagator.add_param("v2", data.shape, data.dtype)
        propagator.add_param("v3", data.shape, data.dtype)
        propagator.add_param("v4", data.shape, data.dtype)
        f = propagator.cfunction

        propagator_oi = propagator.get_kernel_oi(dtype)
        hand_oi = (mul+add)/((load+store)*np.dtype(dtype).itemsize)

        arr = np.empty_like(data)
        f(data, data, data, arr)
        assert(propagator_oi == hand_oi)

    def test_oi_3(self):
        '''OI: (ADD+MUL)/[(LOAD+STORE)*word_size]; word_size: 8(double),4(float)
        Equation:
            v1[i2][i1]
            = (v2[i2][i1] + 2.5F*v2[i2][i1 - 2] + 5*v2[i2][i1 - 1]) /
              (v3[i2][i1] + (1.0F/4.0F)*v3[i2][i1 - 2] + (1.0F/2.0F)*v3[i2][i1 - 1] +
              7.0e-1F*v4[i2][i1] - 1.5e-1F*v4[i2][i1 - 2] - 3.33e-1F*v4[i2][i1 - 1]);
        '''
        load = 4.0
        store = 1.0
        add = 7.0
        mul = 8.0
        dtype = np.float32
        i1, i2 = symbols('i1 i2')
        data = np.arange(100, dtype=np.float32).reshape((10, 10))
        v1 = IndexedBase('v1')
        v2 = IndexedBase('v2')
        v3 = IndexedBase('v3')
        v4 = IndexedBase('v4')
        eq = Eq(v1[i2, i1], (v2[i2, i1] + 5*v2[i2, i1-1] + 2.5*v2[i2, i1-2])
                / ((0.7*v4[i2, i1] - 0.333*v4[i2, i1-1] - 0.15*v4[i2, i1-2])
                   + v3[i2, i1] + v3[i2, i1-1] / 2 + v3[i2, i1-2] / 4))

        propagator = Propagator("process", 10, (10,), 2)
        propagator.stencils = (eq, )
        propagator.add_param("v1", data.shape, data.dtype)
        propagator.add_param("v2", data.shape, data.dtype)
        propagator.add_param("v3", data.shape, data.dtype)
        propagator.add_param("v4", data.shape, data.dtype)
        f = propagator.cfunction

        propagator_oi = propagator.get_kernel_oi(dtype)
        hand_oi = (mul+add)/((load+store)*np.dtype(dtype).itemsize)

        arr = np.empty_like(data)
        f(data, data, data, arr)
        assert(propagator_oi == hand_oi)
