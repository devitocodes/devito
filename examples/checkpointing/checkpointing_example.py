from math import floor
from cached_property import cached_property

from devito import TimeFunction, silencio
from examples.seismic.acoustic import GradientOperator
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from examples.seismic.acoustic.gradient_example import GradientExample
from pyrevolve import Revolver


class CheckpointingExample(GradientExample):
    @cached_property
    def forward_field(self):
        return TimeFunction(name="u", grid=self.model.grid, time_order=2,
                            space_order=self.space_order)

    @cached_property
    def forward_operator(self):
        # Verify uses a forward operator with save=False
        return self.verify_operator

    @cached_property
    def gradient_operator(self):
        return GradientOperator(self.model, self.src, self.rec_g,
                                kernel=self.kernel, spc_order=self.space_order,
                                save=False)

    def gradient(self, m0, maxmem=None):
        cp = DevitoCheckpoint([self.forward_field])
        n_checkpoints = None
        if maxmem is not None:
            n_checkpoints = int(floor(maxmem * 10**6 /
                                      (cp.size * self.forward_field.data.itemsize)))

        wrap_fw = CheckpointOperator(self.forward_operator, u=self.forward_field,
                                     rec=self.rec, m=m0, src=self.src, dt=self.dt)
        wrap_rev = CheckpointOperator(self.gradient_operator, u=self.forward_field,
                                      v=self.adjoint_field, m=m0, rec=self.rec_g,
                                      grad=self.grad, dt=self.dt)
        # Subtracting 2 in the following line assuming time_order=2
        wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, self.src._time_range.num-2)

        wrp.apply_forward()

        self.rec_g.data[:] = self.rec.data[:] - self.rec_t.data[:]

        wrp.apply_reverse()

        # The result is in grad
        return self.grad.data, self.rec


@silencio(log_level='WARNING')
def run(shape=(150, 150), tn=None, spacing=None, kernel='OT2', space_order=4, nbpml=10,
        maxmem=None):
    example = CheckpointingExample(shape, spacing, tn, kernel, space_order, nbpml)
    m0, dm = example.initial_estimate()
    gradient, rec = example.gradient(m0, maxmem)
    example.verify(m0, gradient, rec, dm)


if __name__ == "__main__":
    run(shape=(150, 150), spacing=(15.0, 15.0), tn=750.0, kernel='OT2', space_order=4)
