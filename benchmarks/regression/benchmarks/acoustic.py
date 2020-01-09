from devito import Grid, TimeFunction, Eq, Operator
from examples.seismic.acoustic import AcousticWaveSolver


class IsotropicAcoustic(object):

    params = ([(50, 50, 50)], [4])
    param_names = ['shape', 'space_order']

    repeat = 3

    def time_run(self, shape, space_order):

        #from examples.seismic.acoustic.acoustic_example import run
        #run(shape=shape, space_order=space_order)

        grid = Grid(shape=(4, 4, 4))

        f = TimeFunction(name='f', grid=grid)

        eq = Eq(f.forward, f + 1)

        op = Operator(eq)

        op.apply(time_M=10)
