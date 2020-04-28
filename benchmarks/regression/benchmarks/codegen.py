from examples.seismic.tti.tti_example import tti_setup


repeat = 3


class TTI(object):

    # ASV config
    repeat = 1
    timeout = 600.0

    space_order = 12

    def setup(self):
        self.solver = tti_setup(space_order=TTI.space_order)

    def time_forward(self):
        self.solver.op_fwd()

    def time_adjoint(self):
        self.solver.op_adj()
