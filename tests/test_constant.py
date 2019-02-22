import numpy as np
import pytest

from conftest import skipif
from devito import Grid, Constant, Function, TimeFunction, Eq, solve, Operator

pytestmark = skipif(['yask', 'ops'])

class TestConst(object):
    """
    Class for testing symbolic coefficients functionality
    """

    def test_const_change(self):
        """
        Test that the default replacement rules return the same
        as standard FD.
        """
        
        from IPython import embed, os

        n = 5
        t = Constant(name='t', dtype=np.int32)

        grid = Grid(shape=(2, 2))
        x, y = grid.dimensions
        
        f = TimeFunction(name='f', grid=grid, save=n+1)
        #check = Function(name='check', grid=grid)

        #f.data[:] = 0
        eq = Eq(f.dt-1)
        #eq_test = Eq(check,f[t,x,y])

        stencil = solve(eq, f.forward)
        #print(stencil)
        embed()
        op = Operator([stencil])
        
        #op_test = Operator([eq_test])

        #
        #for j in range(0,n):
            ##if j>0:
                ##t.data = j-1
                ##op_check.apply()
                ##assert(np.amax(check.data[:], axis=None) == j-1)
                ##assert(np.amin(check.data[:], axis=None) == j-1)
            #op.apply(time_m=j, time_M=j, dt=1)
            #embed()
