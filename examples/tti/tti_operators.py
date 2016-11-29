from sympy import *

from devito.dimension import x, y, z
from devito.finite_difference import centered, first_derivative, left
from devito.interfaces import DenseData, TimeData
from devito.operator import Operator
from examples.source_type import SourceLike


class ForwardOperator(Operator):
    """
    Class to setup the forward modelling operator in an acoustic media

    :param model: IGrid() object containing the physical parameters
    :param src: None ot IShot() (not currently supported properly)
    :param damp: Dampening coeeficents for the ABCs
    :param data: IShot() object containing the acquisition geometry and field data
    :param: time_order: Time discretization order
    :param: spc_order: Space discretization order
    :param: trigonometry : COS/SIN functions choice. The default is to use C functions
    `Bhaskara` uses a rational approximation.
    """
    def __init__(self, model, src, damp, data, time_order=2, spc_order=4, save=False,
                 trigonometry='Bhaskara', **kwargs):
        nt, nrec = data.shape
        nt, nsrc = src.shape
        dt = model.get_critical_dt()
        u = TimeData(name="u", shape=model.get_shape_comp(),
                     time_dim=nt, time_order=time_order,
                     space_order=spc_order,
                     save=save, dtype=damp.dtype)
        v = TimeData(name="v", shape=model.get_shape_comp(),
                     time_dim=nt, time_order=time_order,
                     space_order=spc_order,
                     save=save, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(),
                      dtype=damp.dtype)
        m.data[:] = model.padm()

        parm = [m, damp, u, v]

        if model.epsilon is not None:
            epsilon = DenseData(name="epsilon", shape=model.get_shape_comp(),
                                dtype=damp.dtype)
            epsilon.data[:] = model.pad(model.epsilon)
            parm += [epsilon]
        else:
            epsilon = 1.0

        if model.delta is not None:
            delta = DenseData(name="delta", shape=model.get_shape_comp(),
                              dtype=damp.dtype)
            delta.data[:] = model.pad(model.delta)
            parm += [delta]
        else:
            delta = 1.0

        if model.theta is not None:
            theta = DenseData(name="theta", shape=model.get_shape_comp(),
                              dtype=damp.dtype)
            theta.data[:] = model.pad(model.theta)
            parm += [theta]
        else:
            theta = 0

        if model.phi is not None:
            phi = DenseData(name="phi", shape=model.get_shape_comp(),
                            dtype=damp.dtype)
            phi.data[:] = model.pad(model.phi)
            parm += [phi]
        else:
            phi = 0

        u.pad_time = save
        v.pad_time = save
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt,
                         h=model.get_spacing(),
                         coordinates=data.receiver_coords,
                         ndim=len(damp.shape),
                         dtype=damp.dtype,
                         nbpml=model.nbpml)
        source = SourceLike(name="src", npoint=nsrc, nt=nt,
                            dt=dt, h=model.get_spacing(),
                            coordinates=src.receiver_coords,
                            ndim=len(damp.shape),
                            dtype=damp.dtype, nbpml=model.nbpml)
        source.data[:] = src.traces[:]

        def Bhaskarasin(angle):
            if angle == 0:
                return 0
            else:
                return (16.0 * angle * (3.1416 - abs(angle)) /
                        (49.3483 - 4.0 * abs(angle) * (3.1416 - abs(angle))))

        def Bhaskaracos(angle):
            if angle == 0:
                return 1.0
            else:
                return Bhaskarasin(angle + 1.5708)

        s, h = symbols('s h')

        ccos = Bhaskaracos if trigonometry == 'Bhaskara' else cos
        ssin = Bhaskarasin if trigonometry == 'Bhaskara' else sin

        ang0 = ccos(theta)
        ang1 = ssin(theta)
        spc_brd = spc_order


        # Derive stencil from symbolic equation
        if len(m.shape) == 3:
            ang2 = ccos(phi)
            ang3 = ssin(phi)
            Gyp = (ang3 * u.dx - ang2 * u.dyr)
            Gyy = (-first_derivative(Gyp * ang3,
                                     dim=x, side=centered, order=spc_brd) -
                   first_derivative(Gyp * ang2,
                                    dim=y, side=left, order=spc_brd))
            Gyp2 = (ang3 * u.dxr - ang2 * u.dy)
            Gyy2 = (first_derivative(Gyp2 * ang3,
                                     dim=x, side=left, order=spc_brd) +
                    first_derivative(Gyp2 * ang2,
                                     dim=y, side=centered, order=spc_brd))

            Gxp = (ang0 * ang2 * u.dx + ang0 * ang3 * u.dyr - ang1 * u.dzr)
            Gzr = (ang1 * ang2 * v.dx + ang1 * ang3 * v.dyr + ang0 * v.dzr)
            Gxx = (-first_derivative(Gxp * ang0 * ang2,
                                     dim=x, side=centered, order=spc_brd) +
                   first_derivative(Gxp * ang0 * ang3,
                                    dim=y, side=left, order=spc_brd) -
                   first_derivative(Gxp * ang1, dim=z, side=left, order=spc_brd))
            Gzz = (-first_derivative(Gzr * ang1 * ang2,
                                     dim=x, side=centered, order=spc_brd) +
                   first_derivative(Gzr * ang1 * ang3,
                                    dim=y, side=left, order=spc_brd) +
                   first_derivative(Gzr * ang0,
                                    dim=z, side=left, order=spc_brd))
            Gxp2 = (ang0 * ang2 * u.dxr + ang0 * ang3 * u.dy - ang1 * u.dz)
            Gzr2 = (ang1 * ang2 * v.dxr + ang1 * ang3 * v.dy + ang0 * v.dz)
            Gxx2 = (first_derivative(Gxp2 * ang0 * ang2, dim=x, side=left, order=spc_brd) -
                    first_derivative(Gxp2 * ang0 * ang3,
                                     dim=y, side=centered, order=spc_brd) +
                    first_derivative(Gxp2 * ang1,
                                     dim=z, side=centered, order=spc_brd))
            Gzz2 = (first_derivative(Gzr2 * ang1 * ang2,
                                     dim=x, side=left, order=spc_brd) -
                    first_derivative(Gzr2 * ang1 * ang3,
                                     dim=y, side=centered, order=spc_brd) -
                    first_derivative(Gzr2 * ang0,
                                     dim=z, side=centered, order=spc_brd))
            Hp = -(.5*Gxx + .5*Gxx2 + .5*Gyy + .5*Gyy2)
            Hzr = -(.5*Gzz + .5 * Gzz2)

        else:
            Gx1p = (ang0 * u.dxr - ang1 * u.dy)
            Gz1r = (ang1 * v.dxr + ang0 * v.dy)
            Gxx1 = (first_derivative(Gx1p * ang0, dim=x,
                                     side=left, order=spc_brd) +
                    first_derivative(Gx1p * ang1, dim=y,
                                     side=centered, order=spc_brd))
            Gzz1 = (first_derivative(Gz1r * ang1, dim=x,
                                     side=left, order=spc_brd) -
                    first_derivative(Gz1r * ang0, dim=y,
                                     side=centered, order=spc_brd))
            Gx2p = (ang0 * u.dx - ang1 * u.dyr)
            Gz2r = (ang1 * v.dx + ang0 * v.dyr)
            Gxx2 = (-first_derivative(Gx2p * ang0, dim=x,
                    side=centered, order=spc_brd) -
                    first_derivative(Gx2p * ang1, dim=y,
                    side=left, order=spc_brd))
            Gzz2 = (-first_derivative(Gz2r * ang1, dim=x,
                    side=centered, order=spc_brd) +
                    first_derivative(Gz2r * ang0, dim=y,
                    side=left, order=spc_brd))

            Hp = -(.5 * Gxx1 + .5 * Gxx2)
            Hzr = -(.5 * Gzz1 + .5 * Gzz2)

        stencilp = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * u + (s * damp - 2.0 * m) *
             u.backward + 2.0 * s**2 * (epsilon * Hp + delta * Hzr))
        stencilr = 1.0 / (2.0 * m + s * damp) * \
            (4.0 * m * v + (s * damp - 2.0 * m) *
             v.backward + 2.0 * s**2 * (delta * Hp + Hzr))

        # Add substitutions for spacing (temporal and spatial)
        subs = [{s: dt, h: model.get_spacing()}, {s: dt, h: model.get_spacing()}]
        first_stencil = Eq(u.forward, stencilp)
        second_stencil = Eq(v.forward, stencilr)
        stencils = [first_stencil, second_stencil]
        super(ForwardOperator, self).__init__(nt, m.shape,
                                              stencils=stencils,
                                              subs=subs,
                                              spc_border=spc_order,
                                              time_order=time_order,
                                              forward=True,
                                              dtype=m.dtype,
                                              input_params=parm,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [source, source.coordinates, rec, rec.coordinates]
        self.output_params += [v, rec]
        self.propagator.time_loop_stencils_a = (source.add(m, u) + source.add(m, v) +
                                                rec.read2(u, v))
        self.propagator.add_devito_param(source)
        self.propagator.add_devito_param(source.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
