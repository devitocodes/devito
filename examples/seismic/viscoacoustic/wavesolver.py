from devito import VectorTimeFunction, TimeFunction, NODE
from devito.tools import memoized_meth
from examples.seismic import PointSource
from examples.seismic.viscoacoustic.operators import (ForwardOperator, AdjointOperator)


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
    kernel : selects a visco-acoustic equation from the options below:
                'sls' (Standard Linear Solid) :
                1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
                viscoacoustic equation
                2nd order - Bai et al. (2014) viscoacoustic equation
                'ren' - Ren et al. (2014) viscoacoustic equation
                'deng_mcmechan' - Deng and McMechan (2007) viscoacoustic equation
                Defaults to 'sls' 2nd order.
    """
    def __init__(self, model, geometry, space_order=4, kernel='sls', time_order=2,
                 **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="mask")
        self.geometry = geometry

        self.space_order = space_order
        self.kernel = kernel
        self.time_order = time_order
        self._kwargs = kwargs

    @property
    def dt(self):
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, kernel=self.kernel,
                               time_order=self.time_order, **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               space_order=self.space_order, kernel=self.kernel,
                               time_order=self.time_order, **self._kwargs)

    def forward(self, src=None, rec=None, v=None, r=None, p=None, qp=None, b=None,
                vp=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        v : VectorTimeFunction, optional
            The computed particle velocity.
        r : TimeFunction, optional
            The computed memory variable.
        p : TimeFunction, optional
            Stores the computed wavefield.
        qp : Function, optional
            The P-wave quality factor.
        b : Function, optional
            The time-constant inverse density.
        vp : Function or float, optional
            The time-constant velocity.
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.

        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src

        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create all the fields v, p, r
        save_t = src.nt if save else None

        if self.time_order == 1:
            v = v or VectorTimeFunction(name="v", grid=self.model.grid, save=save_t,
                                        time_order=self.time_order,
                                        space_order=self.space_order)
            kwargs.update({k.name: k for k in v})

        # Create the forward wavefield if not provided
        p = p or TimeFunction(name="p", grid=self.model.grid, save=save_t,
                              time_order=self.time_order, space_order=self.space_order,
                              staggered=NODE)

        # Memory variable:
        r = r or TimeFunction(name="r", grid=self.model.grid, save=save_t,
                              time_order=self.time_order, space_order=self.space_order,
                              staggered=NODE)

        # Pick physical parameters from model unless explicitly provided
        b = b or self.model.b
        qp = qp or self.model.qp

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        if self.kernel == 'sls':
            # Execute operator and return wavefield and receiver data
            # With Memory variable
            summary = self.op_fwd(save).apply(src=src, rec=rec, qp=qp, r=r,
                                              p=p, b=b, vp=vp,
                                              dt=kwargs.pop('dt', self.dt), **kwargs)
        else:
            # Execute operator and return wavefield and receiver data
            # Without Memory variable
            summary = self.op_fwd(save).apply(src=src, rec=rec, qp=qp, p=p,
                                              b=b, vp=vp,
                                              dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, p, v, summary

    def adjoint(self, rec, srca=None, va=None, pa=None, vp=None, qp=None, b=None, r=None,
                **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        rec : SparseTimeFunction or array-like
            The receiver data. Please note that
            these act as the source term in the adjoint run.
        srca : SparseTimeFunction or array-like
            The resulting data for the interpolated at the
            original source location.
        va : VectorTimeFunction, optional
            The computed particle velocity.
        pa : TimeFunction, optional
            Stores the computed wavefield.
        vp : Function or float, optional
            The time-constant velocity.
        qp : Function, optional
            The P-wave quality factor.
        b : Function, optional
            The time-constant inverse density.
        r : TimeFunction, optional
            The computed memory variable.

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or PointSource(name='srca', grid=self.model.grid,
                                   time_range=self.geometry.time_axis,
                                   coordinates=self.geometry.src_positions)

        if self.time_order == 1:
            va = va or VectorTimeFunction(name="va", grid=self.model.grid,
                                          time_order=self.time_order,
                                          space_order=self.space_order)
            kwargs.update({k.name: k for k in va})

        pa = pa or TimeFunction(name="pa", grid=self.model.grid,
                                time_order=self.time_order, space_order=self.space_order,
                                staggered=NODE)

        # Memory variable:
        r = r or TimeFunction(name="r", grid=self.model.grid, time_order=self.time_order,
                              space_order=self.space_order, staggered=NODE)

        b = b or self.model.b
        qp = qp or self.model.qp

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        # Execute operator and return wavefield and receiver data
        if self.kernel == 'sls':
            # Execute operator and return wavefield and receiver data
            # With Memory variable
            summary = self.op_adj().apply(src=srca, rec=rec, pa=pa, r=r, b=b, vp=vp,
                                          qp=qp, dt=kwargs.pop('dt', self.dt), **kwargs)
        else:
            summary = self.op_adj().apply(src=srca, rec=rec, pa=pa, vp=vp, b=b, qp=qp,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, pa, va, summary
