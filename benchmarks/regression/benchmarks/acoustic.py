from devito import configuration
from examples.seismic.acoustic.acoustic_example import acoustic_setup


class IsotropicAcoustic(object):

    params = ([(492, 492, 492)], [12])
    param_names = ['shape', 'space_order']

    repeat = 3

    timeout = 600.0

    # Default shape for loop blocking
    x0_blk0_size = 16
    y0_blk0_size = 16

    # Default number of threads -- run across all sockets currently
    nthreads = configuration['platform'].cores_physical

    def time_forward(self, shape, space_order):
        solver = acoustic_setup(shape=shape,
                                space_order=space_order,
                                opt=('advanced', {'openmp': True}))

        solver.forward(x0_blk0_size=IsotropicAcoustic.x0_blk0_size,
                       y0_blk0_size=IsotropicAcoustic.y0_blk0_size,
                       nthreads=IsotropicAcoustic.nthreads,
                       time_M=50)
