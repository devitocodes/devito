import os

import numpy as np
from sympy import sin, Abs

from examples.seismic.utils import scipy_smooth
from devito import (Grid, SubDomain, Function, Constant, warning, mmin, mmax,
                    SubDimension, Eq, Inc, Operator, switchconfig)
from devito.tools import as_tuple

__all__ = ['Model', 'ModelElastic', 'ModelViscoelastic', 'demo_model']


def demo_model(preset, **kwargs):
    """
    Utility function to create preset `Model` objects for
    demonstration and testing purposes. The particular presets are ::

    * `constant-isotropic` : Constant velocity (1.5 km/sec) isotropic model
    * `constant-tti` : Constant anisotropic model. Velocity is 1.5 km/sec and
                      Thomsen parameters are epsilon=.3, delta=.2, theta = .7rad
                      and phi=.35rad for 3D. 2d/3d is defined from the input shape
    * 'layers-isotropic': Simple two-layer model with velocities 1.5 km/s
                 and 2.5 km/s in the top and bottom layer respectively.
                 2d/3d is defined from the input shape
    * 'layers-tti': Simple two-layer TTI model with velocities 1.5 km/s
                    and 2.5 km/s in the top and bottom layer respectively.
                    Thomsen parameters in the top layer are 0 and in the lower layer
                    are epsilon=.3, delta=.2, theta = .5rad and phi=.1 rad for 3D.
                    2d/3d is defined from the input shape
    * 'circle-isotropic': Simple camembert model with velocities 1.5 km/s
                 and 2.5 km/s in a circle at the center. 2D only.
    * 'marmousi2d-isotropic': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``opesci/data`` repository
                    to be available on your machine.
    * 'marmousi2d-tti': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``opesci/data`` repository
                    to be available on your machine.
    * 'marmousi3d-tti': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``opesci/data`` repository
                    to be available on your machine.
    """
    space_order = kwargs.pop('space_order', 2)

    if preset.lower() in ['constant-elastic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        nbpml = kwargs.pop('nbpml', 10)
        dtype = kwargs.pop('dtype', np.float32)
        vp = kwargs.pop('vp', 1.5)
        vs = 0.5 * vp
        rho = 1.0

        return ModelElastic(space_order=space_order, vp=vp, vs=vs, rho=rho, origin=origin,
                            shape=shape, dtype=dtype, spacing=spacing, nbpml=nbpml,
                            **kwargs)

    if preset.lower() in ['constant-viscoelastic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 2.2 km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        nbpml = kwargs.pop('nbpml', 10)
        dtype = kwargs.pop('dtype', np.float32)
        vp = kwargs.pop('vp', 2.2)
        qp = kwargs.pop('qp', 100.)
        vs = kwargs.pop('vs', 1.2)
        qs = kwargs.pop('qs', 70.)
        rho = 2.0

        return ModelViscoelastic(space_order=space_order, vp=vp, qp=qp, vs=vs,
                                 qs=qs, rho=rho, origin=origin, shape=shape,
                                 dtype=dtype, spacing=spacing, nbpml=nbpml,
                                 **kwargs)

    if preset.lower() in ['constant-isotropic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        nbpml = kwargs.pop('nbpml', 10)
        dtype = kwargs.pop('dtype', np.float32)
        vp = kwargs.pop('vp', 1.5)

        return Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, **kwargs)

    elif preset.lower() in ['constant-tti']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        nbpml = kwargs.pop('nbpml', 10)
        dtype = kwargs.pop('dtype', np.float32)
        v = np.empty(shape, dtype=dtype)
        v[:] = 1.5
        epsilon = .3*np.ones(shape, dtype=dtype)
        delta = .2*np.ones(shape, dtype=dtype)
        theta = .7*np.ones(shape, dtype=dtype)
        phi = None
        if len(shape) > 2:
            phi = .35*np.ones(shape, dtype=dtype)

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, phi=phi, **kwargs)

    elif preset.lower() in ['layers-isotropic', 'twolayer-isotropic',
                            '2layer-isotropic']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        dtype = kwargs.pop('dtype', np.float32)
        nbpml = kwargs.pop('nbpml', 10)
        ratio = kwargs.pop('ratio', 3)
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 2.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        v[..., int(shape[-1] / ratio):] = vp_bottom  # Bottom velocity

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, **kwargs)

    elif preset.lower() in ['layers-elastic', 'twolayer-elastic',
                            '2layer-elastic']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        dtype = kwargs.pop('dtype', np.float32)
        nbpml = kwargs.pop('nbpml', 10)
        ratio = kwargs.pop('ratio', 2)
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 2.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        v[..., int(shape[-1] / ratio):] = vp_bottom  # Bottom velocity

        vs = 0.5 * v[:]
        rho = v[:]/vp_top

        return ModelElastic(space_order=space_order, vp=v, vs=vs, rho=rho,
                            origin=origin, shape=shape,
                            dtype=dtype, spacing=spacing, nbpml=nbpml, **kwargs)

    elif preset.lower() in ['layers-viscoelastic', 'twolayer-viscoelastic',
                            '2layer-viscoelastic']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        dtype = kwargs.pop('dtype', np.float32)
        nbpml = kwargs.pop('nbpml', 10)
        ratio = kwargs.pop('ratio', 2)
        vp_top = kwargs.pop('vp_top', 1.6)
        qp_top = kwargs.pop('qp_top', 40.)
        vs_top = kwargs.pop('vs_top', 0.4)
        qs_top = kwargs.pop('qs_top', 30.)
        rho_top = kwargs.pop('rho_top', 1.3)
        vp_bottom = kwargs.pop('vp_bottom', 2.2)
        qp_bottom = kwargs.pop('qp_bottom', 100.)
        vs_bottom = kwargs.pop('vs_bottom', 1.2)
        qs_bottom = kwargs.pop('qs_bottom', 70.)
        rho_bottom = kwargs.pop('qs_bottom', 2.0)

        # Define a velocity profile in km/s
        vp = np.empty(shape, dtype=dtype)
        qp = np.empty(shape, dtype=dtype)
        vs = np.empty(shape, dtype=dtype)
        qs = np.empty(shape, dtype=dtype)
        rho = np.empty(shape, dtype=dtype)
        # Top and bottom P-wave velocity
        vp[:] = vp_top
        vp[..., int(shape[-1] / ratio):] = vp_bottom
        # Top and bottom P-wave quality factor
        qp[:] = qp_top
        qp[..., int(shape[-1] / ratio):] = qp_bottom
        # Top and bottom S-wave velocity
        vs[:] = vs_top
        vs[..., int(shape[-1] / ratio):] = vs_bottom
        # Top and bottom S-wave quality factor
        qs[:] = qs_top
        qs[..., int(shape[-1] / ratio):] = qs_bottom
        # Top and bottom density
        rho[:] = rho_top
        rho[..., int(shape[-1] / ratio):] = rho_bottom

        return ModelViscoelastic(space_order=space_order, vp=vp, qp=qp,
                                 vs=vs, qs=qs, rho=rho, origin=origin,
                                 shape=shape, dtype=dtype, spacing=spacing,
                                 nbpml=nbpml, **kwargs)

    elif preset.lower() in ['layers-tti', 'twolayer-tti', '2layer-tti']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.\
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        dtype = kwargs.pop('dtype', np.float32)
        nbpml = kwargs.pop('nbpml', 10)
        ratio = kwargs.pop('ratio', 2)
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 2.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        v[..., int(shape[-1] / ratio):] = vp_bottom  # Bottom velocity

        epsilon = scipy_smooth(.3*(v - 1.5))
        delta = scipy_smooth(.2*(v - 1.5))
        theta = scipy_smooth(.5*(v - 1.5))
        phi = None
        if len(shape) > 2:
            phi = scipy_smooth(.25*(v - 1.5), shape)

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, phi=phi, **kwargs)

    elif preset.lower() in ['layers-tti-noazimuth', 'twolayer-tti-noazimuth',
                            '2layer-tti-noazimuth']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.\
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        dtype = kwargs.pop('dtype', np.float32)
        nbpml = kwargs.pop('nbpml', 10)
        ratio = kwargs.pop('ratio', 2)
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 2.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        v[..., int(shape[-1] / ratio):] = vp_bottom  # Bottom velocity

        epsilon = .3*(v - 1.5)
        delta = .2*(v - 1.5)
        theta = .5*(v - 1.5)

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, **kwargs)

    elif preset.lower() in ['circle-isotropic']:
        # A simple circle in a 2D domain with a background velocity.
        # By default, the circle velocity is 2.5 km/s,
        # and the background veloity is 3.0 km/s.
        dtype = kwargs.pop('dtype', np.float32)
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        nbpml = kwargs.pop('nbpml', 10)
        vp = kwargs.pop('vp', 3.0)
        vp_background = kwargs.pop('vp_background', 2.5)
        r = kwargs.pop('r', 15)

        # Only a 2D preset is available currently
        assert(len(shape) == 2)

        v = np.empty(shape, dtype=dtype)
        v[:] = vp_background

        a, b = shape[0] / 2, shape[1] / 2
        y, x = np.ogrid[-a:shape[0]-a, -b:shape[1]-b]
        v[x*x + y*y <= r*r] = vp

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, **kwargs)

    elif preset.lower() in ['marmousi-isotropic', 'marmousi2d-isotropic']:
        shape = (1601, 401)
        spacing = (7.5, 7.5)
        origin = (0., 0.)
        nbpml = kwargs.pop('nbpml', 20)

        # Read 2D Marmousi model from opesc/data repo
        data_path = kwargs.get('data_path', None)
        if data_path is None:
            raise ValueError("Path to opesci/data not found! Please specify with "
                             "'data_path=<path/to/opesci/data>'")
        path = os.path.join(data_path, 'Simple2D/vp_marmousi_bi')
        v = np.fromfile(path, dtype='float32', sep="")
        v = v.reshape(shape)

        # Cut the model to make it slightly cheaper
        v = v[301:-300, :]

        return Model(space_order=space_order, vp=v, origin=origin, shape=v.shape,
                     dtype=np.float32, spacing=spacing, nbpml=nbpml, **kwargs)

    elif preset.lower() in ['marmousi-elastic', 'marmousi2d-elastic']:
        shape = (1601, 401)
        spacing = (7.5, 7.5)
        origin = (0., 0.)

        # Read 2D Marmousi model from opesc/data repo
        data_path = kwargs.get('data_path', None)
        if data_path is None:
            raise ValueError("Path to opesci/data not found! Please specify with "
                             "'data_path=<path/to/opesci/data>'")
        path = os.path.join(data_path, 'Simple2D/vp_marmousi_bi')
        v = np.fromfile(path, dtype='float32', sep="")
        v = v.reshape(shape)

        # Cut the model to make it slightly cheaper
        v = v[301:-300, :]
        vs = .5 * v[:]
        rho = v[:]/mmax(v[:])

        return ModelElastic(space_order=space_order, vp=v, vs=vs, rho=rho,
                            origin=origin, shape=v.shape,
                            dtype=np.float32, spacing=spacing, nbpml=20)

    elif preset.lower() in ['marmousi-tti2d', 'marmousi2d-tti']:

        shape_full = (201, 201, 70)
        shape = (201, 70)
        spacing = (10., 10.)
        origin = (0., 0.)
        nbpml = kwargs.pop('nbpml', 20)

        # Read 2D Marmousi model from opesc/data repo
        data_path = kwargs.pop('data_path', None)
        if data_path is None:
            raise ValueError("Path to opesci/data not found! Please specify with "
                             "'data_path=<path/to/opesci/data>'")
        path = os.path.join(data_path, 'marmousi3D/vp_marmousi_bi')

        # velocity
        vp = 1e-3 * np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiVP.raw'),
                                dtype='float32', sep="")
        vp = vp.reshape(shape_full)
        vp = vp[101, :, :]
        # Epsilon, in % in file, resale between 0 and 1
        epsilon = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiEps.raw'),
                              dtype='float32', sep="") * 1e-2
        epsilon = epsilon.reshape(shape_full)
        epsilon = epsilon[101, :, :]
        # Delta, in % in file, resale between 0 and 1
        delta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiDelta.raw'),
                            dtype='float32', sep="") * 1e-2
        delta = delta.reshape(shape_full)
        delta = delta[101, :, :]
        # Theta, in degrees in file, resale in radian
        theta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiTilt.raw'),
                            dtype='float32', sep="")
        theta = np.float32(np.pi / 180 * theta.reshape(shape_full))
        theta = theta[101, :, :]

        return Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                     dtype=np.float32, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, **kwargs)

    elif preset.lower() in ['marmousi-tti3d', 'marmousi3d-tti']:
        shape = (201, 201, 70)
        spacing = (10., 10., 10.)
        origin = (0., 0., 0.)
        nbpml = kwargs.pop('nbpml', 20)

        # Read 2D Marmousi model from opesc/data repo
        data_path = kwargs.pop('data_path', None)
        if data_path is None:
            raise ValueError("Path to opesci/data not found! Please specify with "
                             "'data_path=<path/to/opesci/data>'")
        path = os.path.join(data_path, 'marmousi3D/vp_marmousi_bi')

        # Velcoity
        vp = 1e-3 * np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiVP.raw'),
                                dtype='float32', sep="")
        vp = vp.reshape(shape)
        # Epsilon, in % in file, resale between 0 and 1
        epsilon = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiEps.raw'),
                              dtype='float32', sep="") * 1e-2
        epsilon = epsilon.reshape(shape)
        # Delta, in % in file, resale between 0 and 1
        delta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiDelta.raw'),
                            dtype='float32', sep="") * 1e-2
        delta = delta.reshape(shape)
        # Theta, in degrees in file, resale in radian
        theta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiTilt.raw'),
                            dtype='float32', sep="")
        theta = np.float32(np.pi / 180 * theta.reshape(shape))
        # Phi, in degrees in file, resale in radian
        phi = np.fromfile(os.path.join(data_path, 'marmousi3D/Azimuth.raw'),
                          dtype='float32', sep="")
        phi = np.float32(np.pi / 180 * phi.reshape(shape))

        return Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                     dtype=np.float32, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, phi=phi, **kwargs)

    else:
        raise ValueError("Unknown model preset name")


@switchconfig(log_level='ERROR')
def initialize_damp(damp, nbpml, spacing, mask=False):
    """
    Initialise damping field with an absorbing PML layer.

    Parameters
    ----------
    damp : Function
        The damping field for absorbing boundary condition.
    nbpml : int
        Number of points in the damping layer.
    spacing :
        Grid spacing coefficient.
    mask : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    """
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40)

    eqs = [Eq(damp, 1.0)] if mask else []
    for d in damp.dimensions:
        # left
        dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                  thickness=nbpml)
        pos = Abs((nbpml - (dim_l - d.symbolic_min) + 1) / float(nbpml))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if mask else val
        eqs += [Inc(damp.subs({d: dim_l}), val/d.spacing)]
        # right
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbpml)
        pos = Abs((nbpml - (d.symbolic_max - dim_r) + 1) / float(nbpml))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if mask else val
        eqs += [Inc(damp.subs({d: dim_r}), val/d.spacing)]

    # TODO: Figure out why yask doesn't like it with dse/dle
    Operator(eqs, name='initdamp', dse='noop', dle='noop')()


@switchconfig(log_level='ERROR')
def initialize_function(function, data, nbpml):
    """
    Initialize a `Function` with the given ``data``. ``data``
    does *not* include the PML layers for the absorbing boundary conditions;
    these are added via padding by this function.

    Parameters
    ----------
    function : Function
        The initialised object.
    data : ndarray
        The data array used for initialisation.
    nbpml : int
        Number of PML layers for boundary damping.
    """
    slices = tuple([slice(nbpml, -nbpml, 1) for _ in range(function.grid.dim)])
    function.data[slices] = data
    eqs = []
    for d, s in zip(function.dimensions, function.shape_global):
        dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                  thickness=nbpml)
        eqs += [Eq(function.subs({d: dim_l}), function.subs({d: nbpml}))]
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbpml)
        eqs += [Eq(function.subs({d: dim_r}), function.subs({d: s-nbpml-1}))]

    # TODO: Figure out why yask doesn't like it with dse/dle
    Operator(eqs, name='padfunc', dse='noop', dle='noop')()


class PhysicalDomain(SubDomain):

    name = 'phydomain'

    def __init__(self, nbpml):
        super(PhysicalDomain, self).__init__()
        self.nbpml = nbpml

    def define(self, dimensions):
        return {d: ('middle', self.nbpml, self.nbpml) for d in dimensions}


class GenericModel(object):
    """
    General model class with common properties
    """
    def __init__(self, origin, spacing, shape, space_order, nbpml=20,
                 dtype=np.float32, subdomains=()):
        self.shape = shape
        self.nbpml = int(nbpml)
        self.origin = tuple([dtype(o) for o in origin])

        # Origin of the computational domain with PML to inject/interpolate
        # at the correct index
        origin_pml = tuple([dtype(o - s*nbpml) for o, s in zip(origin, spacing)])
        phydomain = PhysicalDomain(self.nbpml)
        subdomains = subdomains + (phydomain, )
        shape_pml = np.array(shape) + 2 * self.nbpml
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml, dtype=dtype,
                         subdomains=subdomains)

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        known = [getattr(self, i) for i in self._physical_parameters]
        return {i.name: kwargs.get(i.name, i) or i for i in known}

    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def spacing(self):
        """
        Grid spacing for all fields in the physical model.
        """
        return self.grid.spacing

    @property
    def space_dimensions(self):
        """
        Spatial dimensions of the grid
        """
        return self.grid.dimensions

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        return self.grid.spacing_map

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))


class Model(GenericModel):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : array_like or float
        Velocity in km/s.
    nbpml : int, optional
        The number of PML layers for boundary damping.
    dtype : np.float32 or np.float64
        Defaults to 32.
    epsilon : array_like or float, optional
        Thomsen epsilon parameter (0<epsilon<1).
    delta : array_like or float
        Thomsen delta parameter (0<delta<1), delta<epsilon.
    theta : array_like or float
        Tilt angle in radian.
    phi : array_like or float
        Asymuth angle in radian.

    The `Model` provides two symbolic data objects for the
    creation of seismic wave propagation operators:

    m : array_like or float
        The square slowness of the wave.
    damp : Function
        The damping field for absorbing boundary condition.
    """
    def __init__(self, origin, spacing, shape, space_order, vp, nbpml=20,
                 dtype=np.float32, epsilon=None, delta=None, theta=None, phi=None,
                 subdomains=(), **kwargs):
        super(Model, self).__init__(origin, spacing, shape, space_order, nbpml, dtype,
                                    subdomains)

        physical_parameters = []

        # Create square slowness of the wave as symbol `m`
        if isinstance(vp, np.ndarray):
            self._vp = Function(name="vp", grid=self.grid, space_order=space_order)
            initialize_function(self._vp, vp, self.nbpml)
        else:
            self._vp = Constant(name="vp", value=vp)
        self._physical_parameters = ('vp',)
        self._max_vp = np.max(vp)

        # Create dampening field as symbol `damp`
        self.damp = Function(name="damp", grid=self.grid)
        initialize_damp(self.damp, self.nbpml, self.spacing)

        # Additional parameter fields for TTI operators
        self.scale = 1.

        if epsilon is not None:
            if isinstance(epsilon, np.ndarray):
                physical_parameters.append('epsilon')
                self.epsilon = Function(name="epsilon", grid=self.grid)
                initialize_function(self.epsilon, 1 + 2 * epsilon, self.nbpml)
                # Maximum velocity is scale*max(vp) if epsilon > 0
                if mmax(self.epsilon) > 0:
                    self.scale = np.sqrt(mmax(self.epsilon))
            else:
                self.epsilon = 1 + 2 * epsilon
                self.scale = epsilon
        else:
            self.epsilon = 1

        if delta is not None:
            if isinstance(delta, np.ndarray):
                physical_parameters.append('delta')
                self.delta = Function(name="delta", grid=self.grid)
                initialize_function(self.delta, np.sqrt(1 + 2 * delta), self.nbpml)
            else:
                self.delta = delta
        else:
            self.delta = 1

        if theta is not None:
            if isinstance(theta, np.ndarray):
                physical_parameters.append('theta')
                self.theta = Function(name="theta", grid=self.grid,
                                      space_order=space_order)
                initialize_function(self.theta, theta, self.nbpml)
            else:
                self.theta = theta
        else:
            self.theta = 0

        if phi is not None:
            if self.grid.dim < 3:
                warning("2D TTI does not use an azimuth angle Phi, ignoring input")
                self.phi = 0
            elif isinstance(phi, np.ndarray):
                physical_parameters.append('phi')
                self.phi = Function(name="phi", grid=self.grid, space_order=space_order)
                initialize_function(self.phi, phi, self.nbpml)
            else:
                self.phi = phi
        else:
            self.phi = 0

        self._physical_parameters = as_tuple(physical_parameters)

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        dt = self.dtype(coeff * mmin(self.spacing) / (self.scale*self._max_vp))
        return self.dtype("%.3f" % dt)

    @property
    def vp(self):
        """
        `numpy.ndarray` holding the model velocity in km/s.

        Notes
        -----
        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type `Function`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """
        Set a new velocity model and update square slowness.

        Parameters
        ----------
        vp : float or array
            New velocity in km/s.
        """
        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            if vp.shape == self.vp.shape:
                self.vp.data[:] = vp[:]
            elif vp.shape == self.shape:
                initialize_function(self._vp, vp, self.nbpml)
            else:
                raise ValueError("Incorrect input size %s for model of size" % vp.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     self.vp.shape))
        else:
            self._vp.data = vp

        self._max_vp = np.max(vp)

    @property
    def m(self):
        return 1 / (self.vp * self.vp)


class ModelElastic(GenericModel):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats, optional
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : float or array
        P-wave velocity in km/s.
    vs : float or array
        S-wave velocity in km/s.
    nbpml : int, optional
        The number of PML layers for boundary damping.
    rho : float or array, optional
        Density in kg/cm^3 (rho=1 for water).

    The `ModelElastic` provides a symbolic data objects for the
    creation of seismic wave propagation operators:

    damp : Function, optional
        The damping field for absorbing boundary condition.
    """
    def __init__(self, origin, spacing, shape, space_order, vp, vs, rho, nbpml=20,
                 dtype=np.float32):
        super(ModelElastic, self).__init__(origin, spacing, shape, space_order,
                                           nbpml=nbpml, dtype=dtype)

        # Create dampening field as symbol `damp`
        self.damp = Function(name="damp", grid=self.grid)
        initialize_damp(self.damp, self.nbpml, self.spacing, mask=True)

        physical_parameters = []

        self.vp = self._gen_phys_param(vp, 'vp', space_order)
        physical_parameters.append('vp')

        self.vs = self._gen_phys_param(vs, 'vs', space_order)
        physical_parameters.append('vs')

        self.rho = self._gen_phys_param(rho, 'rho', space_order)
        physical_parameters.append('rho')

        self._physical_parameters = as_tuple(physical_parameters)

    def _gen_phys_param(self, field, name, space_order):
        if isinstance(field, np.ndarray):
            function = Function(name=name, grid=self.grid, space_order=space_order)
            initialize_function(function, field, self.nbpml)
        else:
            function = Constant(name=name, value=field)
        return function

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt < h / (sqrt(2) * max(vp)))
        # FIXME: Fix 'Constant' so that that mmax(self.vp) returns the data value
        return self.dtype(.5*mmin(self.spacing) / (np.sqrt(2)*mmax(self.vp.data)))


class ModelViscoelastic(ModelElastic):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats, optional
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : float or array
        P-wave velocity in km/s.
    qp : float or array
        P-wave quality factor (dimensionless).
    vs : float or array
        S-wave velocity in km/s.
    qs : float or array
        S-wave qulaity factor (dimensionless).
    nbpml : int, optional
        The number of PML layers for boundary damping.
    rho : float or array, optional
        Density in kg/cm^3 (rho=1 for water).

    The `ModelElastic` provides a symbolic data objects for the
    creation of seismic wave propagation operators:

    damp : Function, optional
        The damping field for absorbing boundary condition.
    """
    def __init__(self, origin, spacing, shape, space_order, vp, qp, vs, qs, rho,
                 nbpml=20, dtype=np.float32):
        super(ModelViscoelastic, self).__init__(origin, spacing, shape,
                                                space_order, vp, vs, rho,
                                                nbpml=nbpml, dtype=dtype)

        physical_parameters = list(self._physical_parameters)

        self.qp = self._gen_phys_param(qp, 'qp', space_order)
        physical_parameters.append('qp')

        self.qs = self._gen_phys_param(qs, 'qs', space_order)
        physical_parameters.append('qs')

        self._physical_parameters = as_tuple(physical_parameters)

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        # See Blanch, J. O., 1995, "A study of viscous effects in seismic modelling,
        # imaging, and inversion: methodology, computational aspects and sensitivity"
        # for further details:
        # FIXME: Fix 'Constant' so that that mmax(self.vp) returns the data value
        return self.dtype(6.*mmin(self.spacing) /
                          (7.*np.sqrt(self.grid.dim)*mmax(self.vp.data)))
