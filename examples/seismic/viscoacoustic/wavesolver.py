from devito import VectorTimeFunction, TimeFunction, NODE
from devito.tools import memoized_meth
from examples.seismic import Receiver, RickerSource
from examples.seismic.viscoacoustic.operators import ForwardOperator

class ViscoacousticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    equation : selects a visco-acoustic equation from the options below:
                1 - Blanch and Symes (1995) / Dutta and Schuster (2014)
                viscoacoustic equation
                2 - Ren et al. (2014) viscoacoustic equation
                3 - Deng and McMechan (2007) viscoacoustic equation
                Defaults to 1.

    """
    def __init__(self, model, geometry, space_order=4, equation=1, **kwargs):
        self.model = model
        self.geometry = geometry

        self.space_order = space_order
        self.dt = self.model.critical_dt
        self.equation = equation
        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, equation=self.equation,
                               **self._kwargs)

    def forward(self, src=None, rec=None, qp=None, rho=None, v=None, r=None, p=None,
                vp=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (SparseTimeFunction) and
            receivers (SparseTimeFunction) and their position.
        v : VectorTimeFunction, optional
            The computed particle velocity.
        r : TimeFunction, optional
            The computed memory variable.
        p : TimeFunction, optional
            Stores the computed wavefield.
        qp : Function, optional
            The P-wave quality factor.
        rho : Function, optional
            The time-constant density.
        vp : Function or float, optional
            The time-constant velocity.
        save : int or Buffer, optional
            Option to store the entire (unrolled) wavefield.

        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src

        # Create a new receiver object to store the result
        rec = rec or Receiver(name="rec", grid=self.model.grid,
                                time_range=self.geometry.time_axis,
                                coordinates=self.geometry.rec_positions)

        # Create all the fields v, p, r
        save_t = src.nt if save else None
        v = v or VectorTimeFunction(name="v", grid=self.model.grid, save=save_t,
                               time_order=1, space_order=self.space_order)

        # Create the forward wavefield if not provided
        p = p or TimeFunction(name="p", grid=self.model.grid, save=save_t,
                               time_order=1, space_order=self.space_order,
                               staggered=NODE)
        # Memory variable:
        r = r or TimeFunction(name="r", grid=self.model.grid, save=save_t,
                               time_order=1, space_order=self.space_order,
                               staggered=NODE)

        kwargs.update({k.name: k for k in v})

        # Pick physical parameters from model unless explicitly provided
        rho = rho or self.model.rho
        qp = qp or self.model.qp

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        # Pick physical parameters from model
        irho = self.model.irho

        if self.equation == 1:
            # Execute operator and return wavefield and receiver data
            # With Memory variable
            summary = self.op_fwd(save).apply(src=src, rec=rec, qp=qp, rho=rho, r=r,
                                              p=p, irho=irho, vp=vp,
                                              dt=kwargs.pop('dt', self.dt), **kwargs)
        else:
            # Execute operator and return wavefield and receiver data
            # Without Memory variable
            summary = self.op_fwd(save).apply(src=src, rec=rec, qp=qp, rho=rho, p=p,
                                              irho=irho, vp=vp,
                                              dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, p, summary