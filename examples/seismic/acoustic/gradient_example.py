import numpy as np
from numpy import linalg
from cached_property import cached_property

from devito import TimeFunction, Function
from examples.seismic import TimeAxis, Receiver, RickerSource, demo_model
from examples.seismic.acoustic import ForwardOperator, GradientOperator, smooth10


class GradientExample(object):
    def __init__(self, shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0), tn=500.,
                 kernel='OT2', space_order=4, nbpml=10):
        self.kernel = kernel
        self.space_order = space_order
        self._setup_model_and_acquisition(space_order, shape, spacing,
                                          nbpml+int(space_order/2), tn)
        self._true_data()

    @cached_property
    def dt(self):
        v = self.model.critical_dt * (1.73 if self.kernel == 'OT4' else 1.0)
        return self.model.dtype(v)

    def _setup_model_and_acquisition(self, space_order, shape, spacing, nbpml, tn):
        nrec = shape[0]
        model = demo_model('layers-isotropic', space_order=space_order,
                           shape=shape, spacing=spacing, nbpml=nbpml)
        self.model = model
        t0 = 0.0
        time_range = TimeAxis(start=t0, stop=tn, step=self.dt)
        self.nt = time_range.num

        # Define source geometry (center of domain, just below surface)
        src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
        src.coordinates.data[0, :] = np.array(model.domain_size) * .5
        src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

        self.src = src

        # Define receiver geometry (spread across x, just below surface)
        # We need two receiver fields - one for the true (verification) run
        rec_t = Receiver(name='rec', grid=model.grid, time_range=time_range,
                         npoint=nrec)
        rec_t.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0],
                                                   num=nrec)
        rec_t.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

        self.rec_t = rec_t

        # and the other for the smoothed run
        self.rec = Receiver(name='rec', grid=model.grid, time_range=time_range,
                            npoint=nrec, coordinates=rec_t.coordinates.data)

        # Receiver for Gradient
        self.rec_g = Receiver(name="rec", coordinates=self.rec.coordinates.data,
                              grid=model.grid, time_range=time_range)

        # Gradient symbol
        self.grad = Function(name="grad", grid=model.grid)

    def initial_estimate(self):
        m0 = Function(name='m0', grid=self.model.m.grid, space_order=self.space_order)
        m0.data[:] = smooth10(self.model.m.data, self.model.shape_domain)
        dm = np.float32(self.model.m.data - m0.data)
        return m0, dm

    def _true_data(self):
        # Calculate receiver data for true velocity
        self.verify_operator.apply(u=self.temp_field, rec=self.rec_t, src=self.src,
                                   dt=self.dt)

    @cached_property
    def forward_operator(self):
        return ForwardOperator(self.model, self.src, self.rec_t,
                               kernel=self.kernel, spc_order=self.space_order,
                               save=True)

    @cached_property
    def gradient_operator(self):
        return GradientOperator(self.model, self.src, self.rec_g,
                                kernel=self.kernel, spc_order=self.space_order)

    @cached_property
    def verify_operator(self):
        return ForwardOperator(self.model, self.src, self.rec_t,
                               kernel=self.kernel, spc_order=self.space_order,
                               save=False)

    @property
    def temp_field(self):
        return TimeFunction(name="u", grid=self.model.grid, time_order=2,
                            space_order=self.space_order, save=None)

    @cached_property
    def forward_field(self):
        return TimeFunction(name="u", grid=self.model.grid, save=self.src._time_range.num,
                            time_order=2, space_order=self.space_order)

    @cached_property
    def adjoint_field(self):
        return self.temp_field

    def gradient(self, m0):
        # Smooth velocity
        # This is the pass that needs checkpointing <----
        self.forward_operator.apply(u=self.forward_field, rec=self.rec, m=m0,
                                    src=self.src, dt=self.dt)

        self.rec_g.data[:] = self.rec.data[:] - self.rec_t.data[:]

        # Apply the gradient operator to calculate the gradient
        # This is the pass that requires the checkpointed data
        self.gradient_operator.apply(u=self.forward_field, v=self.adjoint_field, m=m0,
                                     rec=self.rec_g, grad=self.grad, dt=self.dt)

        return self.grad.data, self.rec.data

    def _objective_function_value(self, rec_data):
        return .5*linalg.norm(rec_data - self.rec_t.data)**2

    def verify(self, m0, gradient, rec, dm):
        # Objective function value
        F0 = self._objective_function_value(rec.data)

        # <J^T \delta d, dm>
        G = np.dot(gradient.reshape(-1), dm.reshape(-1))
        # FWI Gradient test
        H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]

        error1 = np.zeros(len(H))
        error2 = np.zeros(len(H))

        for i in range(0, len(H)):
            # Add the perturbation to the model
            def initializer(data):
                data[:] = m0.data + H[i] * dm
            mloc = Function(name='mloc', grid=self.model.m.grid,
                            space_order=self.space_order, initializer=initializer)
            # Set field to zero (we're re-using it)
            self.temp_field.data.fill(0)
            # Receiver data for the new model
            # Results will be in rec_s
            self.verify_operator.apply(u=self.temp_field, rec=self.rec, m=mloc,
                                       src=self.src, dt=self.dt)
            d = self.rec.data
            # First order error Phi(m0+dm) - Phi(m0)
            error1[i] = np.absolute(.5*linalg.norm(d - self.rec_t.data)**2 - F0)
            # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
            error2[i] = np.absolute(.5*linalg.norm(d - self.rec_t.data)**2 - F0 - H[i]
                                    * G)

        # Test slope of the  tests
        p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
        p2 = np.polyfit(np.log10(H), np.log10(error2), 1)

        assert np.isclose(p1[0], 1.0, rtol=0.1)
        assert np.isclose(p2[0], 2.0, rtol=0.1)


def run(shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0), tn=500., kernel='OT2',
        space_order=4, nbpml=10):
    example = GradientExample(shape, spacing, tn, kernel, space_order, nbpml)
    m0, dm = example.initial_estimate()
    gradient, rec = example.gradient(m0)
    example.verify(m0, gradient, rec, dm)


if __name__ == "__main__":
    run(shape=(150, 150), spacing=(15.0, 15.0), tn=750.0, kernel='OT2', space_order=4)
