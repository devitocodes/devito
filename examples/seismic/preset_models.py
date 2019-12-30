import os

import numpy as np

from examples.seismic.model import Model, ModelElastic, ModelViscoelastic

__all__ = ['demo_model']


def demo_model(preset, **kwargs):
    """
    Utility function to create preset `Model` objects for
    demonstration and testing purposes. The particular presets are ::

    * `constant-isotropic` : Constant velocity (1.5 km/sec) isotropic model
    * `constant-tti` : Constant anisotropic model. Velocity is 1.5 km/sec and
                      Thomsen parameters are epsilon=.3, delta=.2, theta = .7rad
                      and phi=.35rad for 3D. 2d/3d is defined from the input shape
    * 'layers-isotropic': Simple n-layered model with velocities ranging from 1.5 km/s
                 to 3.5 km/s in the top and bottom layer respectively.
                 2d/3d is defined from the input shape
    * 'layers-elastic': Simple n-layered model with velocities ranging from 1.5 km/s
                    to 3.5 km/s in the top and bottom layer respectively.
                    Vs is set to .5 vp and 0 in the top layer.
    * 'layers-viscoelastic': Simple two layers viscoelastic model.
    * 'layers-tti': Simple n-layered model with velocities ranging from 1.5 km/s
                    to 3.5 km/s in the top and bottom layer respectively.
                    Thomsen parameters in the top layer are 0 and in the lower layers
                    are scaled versions of vp.
                    2d/3d is defined from the input shape
    * 'circle-isotropic': Simple camembert model with velocities 1.5 km/s
                 and 2.5 km/s in a circle at the center. 2D only.
    * 'marmousi2d-isotropic': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``devitocodes/data`` repository
                    to be available on your machine.
    * 'marmousi2d-tti': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``devitocodes/data`` repository
                    to be available on your machine.
    * 'marmousi3d-tti': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``devitocodes/data`` repository
                    to be available on your machine.
    """
    space_order = kwargs.pop('space_order', 2)
    shape = kwargs.pop('shape', (101, 101))
    spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
    origin = kwargs.pop('origin', tuple([0. for _ in shape]))
    nbl = kwargs.pop('nbl', 10)
    dtype = kwargs.pop('dtype', np.float32)
    vp = kwargs.pop('vp', 1.5)
    nlayers = kwargs.pop('nlayers', 3)

    if preset.lower() in ['constant-elastic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.
        vs = 0.5 * vp
        rho = 1.0

        return ModelElastic(space_order=space_order, vp=vp, vs=vs, rho=rho, origin=origin,
                            shape=shape, dtype=dtype, spacing=spacing, nbl=nbl,
                            **kwargs)

    if preset.lower() in ['constant-viscoelastic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 2.2 km/s.
        qp = kwargs.pop('qp', 100.)
        vs = kwargs.pop('vs', 1.2)
        qs = kwargs.pop('qs', 70.)
        rho = 2.

        return ModelViscoelastic(space_order=space_order, vp=vp, qp=qp, vs=vs,
                                 qs=qs, rho=rho, origin=origin, shape=shape,
                                 dtype=dtype, spacing=spacing, nbl=nbl,
                                 **kwargs)

    if preset.lower() in ['constant-isotropic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.

        return Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbl=nbl, **kwargs)

    elif preset.lower() in ['constant-tti']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5 km/s.
        v = np.empty(shape, dtype=dtype)
        v[:] = 1.5
        epsilon = .3*np.ones(shape, dtype=dtype)
        delta = .2*np.ones(shape, dtype=dtype)
        theta = .7*np.ones(shape, dtype=dtype)
        phi = None
        if len(shape) > 2:
            phi = .35*np.ones(shape, dtype=dtype)

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbl=nbl, epsilon=epsilon,
                     delta=delta, theta=theta, phi=phi, **kwargs)

    elif preset.lower() in ['layers-isotropic']:
        # A n-layers model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 3.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        vp_i = np.linspace(vp_top, vp_bottom, nlayers)
        for i in range(1, nlayers):
            v[..., i*int(shape[-1] / nlayers):] = vp_i[i]  # Bottom velocity

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbl=nbl, **kwargs)

    elif preset.lower() in ['layers-elastic']:
        # A n-layers model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 3.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        vp_i = np.linspace(vp_top, vp_bottom, nlayers)
        for i in range(1, nlayers):
            v[..., i*int(shape[-1] / nlayers):] = vp_i[i]  # Bottom velocity

        vs = 0.5 * v[:]
        rho = 0.31 * (1e3*v)**0.25
        rho[v < 1.51] = 1.0
        vs[v < 1.51] = 0.0

        return ModelElastic(space_order=space_order, vp=v, vs=vs, rho=rho,
                            origin=origin, shape=shape,
                            dtype=dtype, spacing=spacing, nbl=nbl, **kwargs)

    elif preset.lower() in ['layers-viscoelastic', 'twolayer-viscoelastic',
                            '2layer-viscoelastic']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.6 km/s,
        # and the bottom part of the domain has 2.2 km/s.
        ratio = kwargs.pop('ratio', 3)
        vp_top = kwargs.pop('vp_top', 1.6)
        qp_top = kwargs.pop('qp_top', 40.)
        vs_top = kwargs.pop('vs_top', 0.4)
        qs_top = kwargs.pop('qs_top', 30.)
        rho_top = kwargs.pop('rho_top', 1.3)
        vp_bottom = kwargs.pop('vp_bottom', 2.2)
        qp_bottom = kwargs.pop('qp_bottom', 100.)
        vs_bottom = kwargs.pop('vs_bottom', 1.2)
        qs_bottom = kwargs.pop('qs_bottom', 70.)
        rho_bottom = kwargs.pop('qs_bottom', 2.)

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
                                 nbl=nbl, **kwargs)

    elif preset.lower() in ['layers-tti', 'layers-tti-noazimuth']:
        # A n-layers model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.\
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 3.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        vp_i = np.linspace(vp_top, vp_bottom, nlayers)
        for i in range(1, nlayers):
            v[..., i*int(shape[-1] / nlayers):] = vp_i[i]  # Bottom velocity

        epsilon = .3*(v - 1.5)
        delta = .2*(v - 1.5)
        theta = .5*(v - 1.5)
        phi = None
        if len(shape) > 2 and preset.lower() not in ['layers-tti-noazimuth']:
            phi = .25*(v - 1.5)
        model = Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                      dtype=dtype, spacing=spacing, nbl=nbl, epsilon=epsilon,
                      delta=delta, theta=theta, phi=phi, **kwargs)
        if len(shape) > 2 and preset.lower() not in ['layers-tti-noazimuth']:
            model.smooth(('epsilon', 'delta', 'theta', 'phi'))
        else:
            model.smooth(('epsilon', 'delta', 'theta'))

        return model

    elif preset.lower() in ['circle-isotropic']:
        # A simple circle in a 2D domain with a background velocity.
        # By default, the circle velocity is 2.5 km/s,
        # and the background veloity is 3.0 km/s.
        vp = kwargs.pop('vp_circle', 3.0)
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
                     dtype=dtype, spacing=spacing, nbl=nbl, **kwargs)

    elif preset.lower() in ['marmousi-isotropic', 'marmousi2d-isotropic']:
        shape = (1601, 401)
        spacing = (7.5, 7.5)
        origin = (0., 0.)
        nbl = kwargs.pop('nbl', 20)

        # Read 2D Marmousi model from devitocodes/data repo
        data_path = kwargs.get('data_path', None)
        if data_path is None:
            raise ValueError("Path to devitocodes/data not found! Please specify with "
                             "'data_path=<path/to/devitocodes/data>'")
        path = os.path.join(data_path, 'Simple2D/vp_marmousi_bi')
        v = np.fromfile(path, dtype='float32', sep="")
        v = v.reshape(shape)

        # Cut the model to make it slightly cheaper
        v = v[301:-300, :]

        return Model(space_order=space_order, vp=v, origin=origin, shape=v.shape,
                     dtype=np.float32, spacing=spacing, nbl=nbl, **kwargs)

    elif preset.lower() in ['marmousi-tti2d', 'marmousi2d-tti',
                            'marmousi-tti3d', 'marmousi3d-tti']:

        shape_full = (201, 201, 70)
        shape = (201, 70)
        spacing = (10., 10.)
        origin = (0., 0.)
        nbl = kwargs.pop('nbl', 20)

        # Read 2D Marmousi model from devitocodes/data repo
        data_path = kwargs.pop('data_path', None)
        if data_path is None:
            raise ValueError("Path to devitocodes/data not found! Please specify with "
                             "'data_path=<path/to/devitocodes/data>'")
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

        if preset.lower() in ['marmousi-tti3d', 'marmousi3d-tti']:
            # Phi, in degrees in file, resale in radian
            phi = np.fromfile(os.path.join(data_path, 'marmousi3D/Azimuth.raw'),
                              dtype='float32', sep="")
            phi = np.float32(np.pi / 180 * phi.reshape(shape))
        else:
            phi = None

        return Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                     dtype=np.float32, spacing=spacing, nbl=nbl, epsilon=epsilon,
                     delta=delta, theta=theta, phi=phi, **kwargs)

    else:
        raise ValueError("Unknown model preset name")
