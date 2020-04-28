from examples.seismic.acoustic.acoustic_example import acoustic_setup


class IsotropicAcoustic(object):

    # ASV parametrization
    params = ([(492, 492, 492)], [12])
    param_names = ['shape', 'space_order']

    # ASV config
    repeat = 3
    timeout = 600.0

    # Default shape for loop blocking
    x0_blk0_size = 16
    y0_blk0_size = 16

    def setup(self, shape, space_order):
        self.solver = acoustic_setup(shape=shape, space_order=space_order,
                                     opt=('advanced', {'openmp': True}))

    def time_forward(self, shape, space_order):
        self.solver.forward(x0_blk0_size=IsotropicAcoustic.x0_blk0_size,
                            y0_blk0_size=IsotropicAcoustic.y0_blk0_size,
                            time_M=50)
