# coding: utf-8
from devito import Function, TimeFunction, warning, DevitoCheckpoint, CheckpointOperator
from devito.tools import memoized_meth
from examples.seismic.tti.operators import ForwardOperator, AdjointOperator
from examples.seismic.tti.operators import JacobianOperator, JacobianAdjOperator
from examples.seismic.tti.operators import particle_velocity_fields
from pyrevolve import Revolver


class AnisotropicWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.

    Notes
    -----
    space_order must be even and it is recommended to be a multiple of 4
    """
    def __init__(self, model, geometry, space_order=4, kernel='centered',
                 **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry
        self.kernel = kernel

        if space_order % 2 != 0:
            raise ValueError("space_order must be even but got %s"
                             % space_order)

        if space_order % 4 != 0:
            warning("It is recommended for space_order to be a multiple of 4" +
                    "but got %s" % space_order)

        self.space_order = space_order

        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=False):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, kernel=self.kernel,
                               **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               space_order=self.space_order, kernel=self.kernel,
                               **self._kwargs)

    @memoized_meth
    def op_jac(self):
        """Cached operator for born runs"""
        return JacobianOperator(self.model, save=None, geometry=self.geometry,
                                space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_jacadj(self, save=True):
        """Cached operator for gradient runs"""
        return JacobianAdjOperator(self.model, save=save, geometry=self.geometry,
                                   space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec=None, u=None, v=None, model=None,
                save=False, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (SparseTimeFunction) and
            receivers (SparseTimeFunction) and their position.
        u : TimeFunction, optional
            The computed wavefield first component.
        v : TimeFunction, optional
            The computed wavefield second component.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        epsilon : Function or float, optional
            The time-constant first Thomsen parameter.
        delta : Function or float, optional
            The time-constant second Thomsen parameter.
        theta : Function or float, optional
            The time-constant Dip angle (radians).
        phi : Function or float, optional
            The time-constant Azimuth angle (radians).
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.
        kernel : str, optional
            Type of discretization, centered or shifted.

        Returns
        -------
        Receiver, wavefield and performance summary.
        """
        if self.kernel == 'staggered':
            time_order = 1
            dims = self.model.space_dimensions
            stagg_u = (-dims[-1])
            stagg_v = (-dims[0], -dims[1]) if self.model.grid.dim == 3 else (-dims[0])
        else:
            time_order = 2
            stagg_u = stagg_v = None
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        if u is None:
            u = TimeFunction(name='u', grid=self.model.grid, staggered=stagg_u,
                             save=self.geometry.nt if save else None,
                             time_order=time_order,
                             space_order=self.space_order)
        # Create the forward wavefield if not provided
        if v is None:
            v = TimeFunction(name='v', grid=self.model.grid, staggered=stagg_v,
                             save=self.geometry.nt if save else None,
                             time_order=time_order,
                             space_order=self.space_order)

        if self.kernel == 'staggered':
            vx, vz, vy = particle_velocity_fields(self.model, self.space_order)
            kwargs["vx"] = vx
            kwargs["vz"] = vz
            if vy is not None:
                kwargs["vy"] = vy

        model = model or self.model
        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        if self.model.dim < 3:
            kwargs.pop('phi', None)
        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec=rec, u=u, v=v,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, v, summary

    def adjoint(self, rec, srca=None, p=None, r=None, model=None,
                save=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (SparseTimeFunction) and
            receivers (SparseTimeFunction) and their position.
        p : TimeFunction, optional
            The computed wavefield first component.
        r : TimeFunction, optional
            The computed wavefield second component.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        epsilon : Function or float, optional
            The time-constant first Thomsen parameter.
        delta : Function or float, optional
            The time-constant second Thomsen parameter.
        theta : Function or float, optional
            The time-constant Dip angle (radians).
        phi : Function or float, optional
            The time-constant Azimuth angle (radians).

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        if self.kernel == 'staggered':
            time_order = 1
            dims = self.model.space_dimensions
            stagg_p = (-dims[-1])
            stagg_r = (-dims[0], -dims[1]) if self.model.grid.dim == 3 else (-dims[0])
        else:
            time_order = 2
            stagg_p = stagg_r = None

        # Source term is read-only, so re-use the default
        srca = srca or self.geometry.new_src(name='srca', src_type=None)

        # Create the wavefield if not provided
        if p is None:
            p = TimeFunction(name='p', grid=self.model.grid, staggered=stagg_p,
                             time_order=time_order,
                             space_order=self.space_order)
        # Create the wavefield if not provided
        if r is None:
            r = TimeFunction(name='r', grid=self.model.grid, staggered=stagg_r,
                             time_order=time_order,
                             space_order=self.space_order)

        if self.kernel == 'staggered':
            vx, vz, vy = particle_velocity_fields(self.model, self.space_order)
            kwargs["vx"] = vx
            kwargs["vz"] = vz
            if vy is not None:
                kwargs["vy"] = vy

        model = model or self.model
        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        if self.model.dim < 3:
            kwargs.pop('phi', None)
        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(srca=srca, rec=rec, p=p, r=r,
                                      dt=kwargs.pop('dt', self.dt),
                                      time_m=0 if time_order == 1 else None,
                                      **kwargs)
        return srca, p, r, summary

    def jacobian(self, dm, src=None, rec=None, u0=None, v0=None, du=None, dv=None,
                 model=None, save=None, kernel='centered', **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            The computed background wavefield first component.
        v : TimeFunction, optional
            The computed background wavefield second component.
        du : TimeFunction, optional
            The computed perturbed wavefield first component.
        dv : TimeFunction, optional
            The computed perturbed wavefield second component.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        epsilon : Function or float, optional
            The time-constant first Thomsen parameter.
        delta : Function or float, optional
            The time-constant second Thomsen parameter.
        theta : Function or float, optional
            The time-constant Dip angle (radians).
        phi : Function or float, optional
            The time-constant Azimuth angle (radians).
        """
        if kernel != 'centered':
            raise ValueError('Only centered kernel is supported for the jacobian')

        dt = kwargs.pop('dt', self.dt)
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefields u, v du and dv if not provided
        u0 = u0 or TimeFunction(name='u0', grid=self.model.grid,
                                time_order=2, space_order=self.space_order)
        v0 = v0 or TimeFunction(name='v0', grid=self.model.grid,
                                time_order=2, space_order=self.space_order)
        du = du or TimeFunction(name='du', grid=self.model.grid,
                                time_order=2, space_order=self.space_order)
        dv = dv or TimeFunction(name='dv', grid=self.model.grid,
                                time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        if self.model.dim < 3:
            kwargs.pop('phi', None)

        # Execute operator and return wavefield and receiver data
        summary = self.op_jac().apply(dm=dm, u0=u0, v0=v0, du=du, dv=dv, src=src,
                                      rec=rec, dt=dt, **kwargs)
        return rec, u0, v0, du, dv, summary

    def jacobian_adjoint(self, rec, u0, v0, du=None, dv=None, dm=None, model=None,
                         checkpointing=False, kernel='centered', **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u0 : TimeFunction
            The computed background wavefield.
        v0 : TimeFunction, optional
            The computed background wavefield.
        du : Function or float
            The computed perturbed wavefield.
        dv : Function or float
            The computed perturbed wavefield.
        dm : Function, optional
            Stores the gradient field.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        epsilon : Function or float, optional
            The time-constant first Thomsen parameter.
        delta : Function or float, optional
            The time-constant second Thomsen parameter.
        theta : Function or float, optional
            The time-constant Dip angle (radians).
        phi : Function or float, optional
            The time-constant Azimuth angle (radians).

        Returns
        -------
        Gradient field and performance summary.
        """
        if kernel != 'centered':
            raise ValueError('Only centered kernel is supported for the jacobian_adj')

        dt = kwargs.pop('dt', self.dt)
        # Gradient symbol
        dm = dm or Function(name='dm', grid=self.model.grid)

        # Create the perturbation wavefields if not provided
        du = du or TimeFunction(name='du', grid=self.model.grid,
                                time_order=2, space_order=self.space_order)
        dv = dv or TimeFunction(name='dv', grid=self.model.grid,
                                time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        if self.model.dim < 3:
            kwargs.pop('phi', None)

        if checkpointing:
            u0 = TimeFunction(name='u0', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
            v0 = TimeFunction(name='v0', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
            cp = DevitoCheckpoint([u0, v0])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False), src=self.geometry.src,
                                         u=u0, v=v0, dt=dt, **kwargs)
            wrap_rev = CheckpointOperator(self.op_jacadj(save=False), u0=u0, v0=v0,
                                          du=du, dv=dv, rec=rec, dm=dm, dt=dt, **kwargs)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            summary = self.op_jacadj().apply(rec=rec, dm=dm, u0=u0, v0=v0, du=du, dv=dv,
                                             dt=dt, **kwargs)
        return dm, summary
