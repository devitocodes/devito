from devito import Grid, Function, TimeFunction, SparseTimeFunction, Eq, Operator


# ASV config
repeat = 10
timeout = 600.0


class Processing:

    def setup(self):
        grid = Grid(shape=(5, 5, 5))

        funcs = [Function(name='f%d' % n, grid=grid) for n in range(30)]
        tfuncs = [TimeFunction(name='u%d' % n, grid=grid) for n in range(30)]
        stfuncs = [SparseTimeFunction(name='su%d' % n, grid=grid, npoint=1, nt=100)
                   for n in range(30)]
        v = TimeFunction(name='v', grid=grid, space_order=2)

        eq = Eq(v.forward, v.laplace + sum(funcs) + sum(tfuncs) + sum(stfuncs),
                subdomain=grid.interior)

        self.op = Operator(eq, opt='noop')

        # Allocate data, populate cached properties, etc.
        self.op.arguments(time_M=98)

    def time_processing(self):
        self.op.arguments(time_M=98)
