import ctypes
import pickle as pickle0

import cloudpickle as pickle1
import pytest
import numpy as np
from sympy import Symbol

from devito import (Constant, Eq, Function, TimeFunction, SparseFunction, Grid,
                    Dimension, SubDimension, ConditionalDimension, IncrDimension,
                    TimeDimension, SteppingDimension, Operator, MPI, Min, solve,
                    PrecomputedSparseTimeFunction, SubDomain)
from devito.ir import Backward, Forward, GuardFactor, GuardBound, GuardBoundNext
from devito.data import LEFT, OWNED
from devito.finite_differences.tools import direct, transpose, left, right, centered
from devito.mpi.halo_scheme import Halo
from devito.mpi.routines import (MPIStatusObject, MPIMsgEnriched, MPIRequestObject,
                                 MPIRegion)
from devito.types import (Array, CustomDimension, Symbol as dSymbol, Scalar,
                          PointerArray, Lock, PThreadArray, SharedData, Timer,
                          DeviceID, NPThreads, ThreadID, TempFunction, Indirection,
                          FIndexed, ComponentAccess)
from devito.types.basic import BoundSymbol, AbstractSymbol
from devito.tools import EnrichedTuple
from devito.symbolics import (IntDiv, ListInitializer, FieldFromPointer,
                              CallFromPointer, DefFunction, Cast, SizeOf,
                              pow_to_mul)
from examples.seismic import (demo_model, AcquisitionGeometry,
                              TimeAxis, RickerSource, Receiver)


class SD(SubDomain):
    name = 'sd'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: x, y: ('middle', 1, 1), z: ('right', 2)}


@pytest.mark.parametrize('pickle', [pickle0, pickle1])
class TestBasic:

    def test_abstractsymbol(self, pickle):
        s0 = AbstractSymbol('s')
        s1 = AbstractSymbol('s', nonnegative=True, integer=False)

        pkl_s0 = pickle.dumps(s0)
        pkl_s1 = pickle.dumps(s1)

        new_s0 = pickle.loads(pkl_s0)
        new_s1 = pickle.loads(pkl_s1)

        assert s0.assumptions0 == new_s0.assumptions0
        assert s1.assumptions0 == new_s1.assumptions0

        assert s0 == new_s0
        assert s1 == new_s1

    def test_constant(self, pickle):
        c = Constant(name='c')
        assert c.data == 0.
        c.data = 1.

        pkl_c = pickle.dumps(c)
        new_c = pickle.loads(pkl_c)

        # .data is initialized, so it should have been pickled too
        assert np.all(c.data == 1.)
        assert np.all(new_c.data == 1.)

    def test_dimension(self, pickle):
        d = Dimension(name='d')

        pkl_d = pickle.dumps(d)
        new_d = pickle.loads(pkl_d)

        assert d.name == new_d.name
        assert d.dtype == new_d.dtype
        assert d.symbolic_min == new_d.symbolic_min
        assert d.symbolic_max == new_d.symbolic_max

    def test_enrichedtuple(self, pickle):
        # Dummy enriched tuple
        tup = EnrichedTuple(11, 31, getters=('a', 'b'), left=[3, 4], right=[5, 6])

        pkl_t = pickle.dumps(tup)
        new_t = pickle.loads(pkl_t)

        assert new_t == tup  # This only tests the actual tuple
        assert new_t.getters == tup.getters
        assert new_t.left == tup.left
        assert new_t.right == tup.right

    def test_enrichedtuple_rebuild(self, pickle):
        tup = EnrichedTuple(11, 31, getters=('a', 'b'), left=[3, 4], right=[5, 6])
        new_t = tup._rebuild()

        assert new_t == tup
        assert new_t.getters == tup.getters
        assert new_t.left == tup.left
        assert new_t.right == tup.right

    @pytest.mark.parametrize('on_sd', [False, True])
    def test_function(self, pickle, on_sd):
        grid = Grid(shape=(3, 3, 3))

        if on_sd:
            sd = SD(grid=grid)
            f = Function(name='f', grid=sd)
        else:
            f = Function(name='f', grid=grid)
        f.data[0] = 1.

        pkl_f = pickle.dumps(f)
        new_f = pickle.loads(pkl_f)

        # .data is initialized, so it should have been pickled too
        assert np.all(f.data[0] == 1.)
        assert np.all(new_f.data[0] == 1.)

        assert f.space_order == new_f.space_order
        assert f.dtype == new_f.dtype
        assert f.shape == new_f.shape

    @pytest.mark.parametrize('interp', ['linear', 'sinc'])
    def test_sparse_function(self, pickle, interp):
        grid = Grid(shape=(3,))
        sf = SparseFunction(name='sf', grid=grid, npoint=3, space_order=2,
                            coordinates=[(0.,), (1.,), (2.,)],
                            interpolation=interp)
        sf.data[0] = 1.

        pkl_sf = pickle.dumps(sf)
        new_sf = pickle.loads(pkl_sf)

        # .data is initialized, so it should have been pickled too
        assert np.all(sf.data[0] == 1.)
        assert np.all(new_sf.data[0] == 1.)
        assert new_sf.interpolation == interp

        # coordinates should also have been pickled
        assert np.all(sf.coordinates.data == new_sf.coordinates.data)

        assert sf.space_order == new_sf.space_order
        assert sf.dtype == new_sf.dtype
        assert sf.npoint == new_sf.npoint

    @pytest.mark.parametrize('mode', ['coordinates', 'gridpoints'])
    def test_precomputed_sparse_function(self, mode, pickle):
        grid = Grid(shape=(11, 11))

        coords = [(0., 0.), (.5, .5), (.7, .2)]
        gridpoints = [(0, 0), (6, 6), (8, 3)]
        keys = {'coordinates': coords, 'gridpoints': gridpoints}
        kw = {mode: keys[mode]}
        othermode = 'coordinates' if mode == 'gridpoints' else 'gridpoints'

        sf = PrecomputedSparseTimeFunction(
            name='sf', grid=grid, r=2, npoint=3, nt=5,
            interpolation_coeffs=np.random.randn(3, 2, 2), **kw
        )
        sf.data[2, 1] = 5.

        pkl_sf = pickle.dumps(sf)
        new_sf = pickle.loads(pkl_sf)

        # .data is initialized, so it should have been pickled too
        assert new_sf.data[2, 1] == 5.

        # gridpoints and interpolation coefficients must have been pickled
        assert np.all(sf.interpolation_coeffs.data == new_sf.interpolation_coeffs.data)

        # coordinates, since they were given, should also have been pickled
        assert np.all(getattr(sf, mode).data == getattr(new_sf, mode).data)
        assert getattr(sf, othermode) is None
        assert getattr(new_sf, othermode) is None

        assert sf._radius == new_sf._radius == 1
        assert sf.space_order == new_sf.space_order
        assert sf.time_order == new_sf.time_order
        assert sf.dtype == new_sf.dtype
        assert sf.npoint == new_sf.npoint == 3

    def test_alias_sparse_function(self, pickle):
        grid = Grid(shape=(3,))
        sf = SparseFunction(name='sf', grid=grid, npoint=3, space_order=2,
                            coordinates=[(0.,), (1.,), (2.,)])
        sf.data[0] = 1.

        # Create alias
        f0 = sf._rebuild(name='f0', alias=True)
        pkl_f0 = pickle.dumps(f0)
        new_f0 = pickle.loads(pkl_f0)

        assert f0.data is None and new_f0.data is None
        assert f0.coordinates.data is None and new_f0.coordinates.data is None

        assert sf.space_order == f0.space_order == new_f0.space_order
        assert sf.dtype == f0.dtype == new_f0.dtype
        assert sf.npoint == f0.npoint == new_f0.npoint

    @pytest.mark.parametrize('interp', ['linear', 'sinc'])
    @pytest.mark.parametrize('op', ['inject', 'interpolate'])
    def test_sparse_op(self, pickle, interp, op):
        grid = Grid(shape=(3,))
        sf = SparseFunction(name='sf', grid=grid, npoint=3, space_order=2,
                            coordinates=[(0.,), (1.,), (2.,)],
                            interpolation=interp)
        u = Function(name='u', grid=grid, space_order=4)

        if op == 'inject':
            expr = sf.inject(u, sf)
        else:
            expr = sf.interpolate(u)

        pkl_expr = pickle.dumps(expr)
        new_expr = pickle.loads(pkl_expr)

        assert new_expr.interpolator._name == expr.interpolator._name
        assert new_expr.implicit_dims == expr.implicit_dims
        assert str(new_expr.evaluate) == str(expr.evaluate)

    def test_internal_symbols(self, pickle):
        s = dSymbol(name='s', dtype=np.float32)
        pkl_s = pickle.dumps(s)
        new_s = pickle.loads(pkl_s)
        assert new_s.name == s.name
        assert new_s.dtype is np.float32

        s = Scalar(name='s', dtype=np.int32, is_const=True)
        pkl_s = pickle.dumps(s)
        new_s = pickle.loads(pkl_s)
        assert new_s.name == s.name
        assert new_s.dtype is np.int32
        assert new_s.is_const is True
        assert new_s.is_nonnegative is None

        s = Scalar(name='s', nonnegative=True)
        pkl_s = pickle.dumps(s)
        new_s = pickle.loads(pkl_s)
        assert new_s.name == s.name
        assert new_s.assumptions0['nonnegative'] is True
        assert new_s.is_nonnegative is True

    def test_bound_symbol(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        f = Function(name='f', grid=grid)

        bs = f._C_symbol
        pkl_bs = pickle.dumps(bs)
        new_bs = pickle.loads(pkl_bs)

        assert isinstance(new_bs, BoundSymbol)
        assert new_bs.name == bs.name
        assert isinstance(new_bs.function, Function)
        assert str(new_bs.function) == str(bs.function)

    def test_indirection(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        f = Function(name='f', grid=grid)

        ind = Indirection(name='ofs', mapped=f)

        pkl_ind = pickle.dumps(ind)
        new_ind = pickle.loads(pkl_ind)

        assert new_ind.name == ind.name
        assert isinstance(new_ind.mapped, Function)
        assert str(new_ind.mapped) == str(ind.mapped) == str(f)
        assert new_ind.dtype == ind.dtype

    def test_array(self, pickle):
        grid = Grid(shape=(3, 3))
        d = Dimension(name='d')

        a = Array(name='a', dimensions=grid.dimensions, dtype=np.int32,
                  halo=((1, 1), (2, 2)), padding=((2, 2), (2, 2)),
                  space='host', scope='stack')

        pkl_a = pickle.dumps(a)
        new_a = pickle.loads(pkl_a)
        assert new_a.name == a.name
        assert new_a.dtype is np.int32
        assert new_a.dimensions[0].name == 'x'
        assert new_a.dimensions[1].name == 'y'
        assert new_a.halo == ((1, 1), (2, 2))
        assert new_a.padding == ((2, 2), (2, 2))
        assert new_a.space == 'host'
        assert new_a.scope == 'stack'

        # Now with a pointer array
        pa = PointerArray(name='pa', dimensions=d, array=a)

        pkl_pa = pickle.dumps(pa)
        new_pa = pickle.loads(pkl_pa)
        assert new_pa.name == pa.name
        assert new_pa.dim.name == 'd'
        assert new_pa.array.name == 'a'

    def test_sub_dimension(self, pickle):
        di = SubDimension.middle('di', Dimension(name='d'), 1, 1)

        pkl_di = pickle.dumps(di)
        new_di = pickle.loads(pkl_di)

        assert di.name == new_di.name
        assert di.dtype == new_di.dtype
        assert di.parent.name == new_di.parent.name
        assert di._thickness == new_di._thickness
        assert di._interval == new_di._interval

    def test_conditional_dimension(self, pickle):
        d = Dimension(name='d')
        s = Scalar(name='s')
        cd = ConditionalDimension(name='ci', parent=d, factor=4, condition=s > 3)

        pkl_cd = pickle.dumps(cd)
        new_cd = pickle.loads(pkl_cd)

        assert cd.name == new_cd.name
        assert cd.parent.name == new_cd.parent.name
        assert cd.factor == new_cd.factor
        assert cd.symbolic_factor == new_cd.symbolic_factor
        assert cd.condition == new_cd.condition

    def test_incr_dimension(self, pickle):
        s = Scalar(name='s')
        d = Dimension(name='d')
        dd = IncrDimension('dd', d, s, 5, 2)

        pkl_dd = pickle.dumps(dd)
        new_dd = pickle.loads(pkl_dd)

        assert dd.name == new_dd.name
        assert dd.parent.name == new_dd.parent.name
        assert dd.symbolic_min == new_dd.symbolic_min
        assert dd.symbolic_max == new_dd.symbolic_max
        assert dd.step == new_dd.step

    def test_custom_dimension(self, pickle):
        symbolic_size = Constant(name='d_custom_size')
        d = CustomDimension(name='d', symbolic_size=symbolic_size)

        pkl_d = pickle.dumps(d)
        new_d = pickle.loads(pkl_d)

        assert d.name == new_d.name
        assert d.symbolic_size.name == new_d.symbolic_size.name

    def test_lock(self, pickle):
        ld = CustomDimension(name='ld', symbolic_size=2)
        lock = Lock(name='lock', dimensions=ld)

        pkl_lock = pickle.dumps(lock)
        new_lock = pickle.loads(pkl_lock)

        lock.name == new_lock.name
        new_lock.dimensions[0].symbolic_size == ld.symbolic_size

    def test_p_thread_array(self, pickle):
        a = PThreadArray(name='threads', npthreads=4)

        pkl_a = pickle.dumps(a)
        new_a = pickle.loads(pkl_a)

        assert a.name == new_a.name
        assert a.dimensions[0].name == new_a.dimensions[0].name
        assert a.size == new_a.size

    def test_shared_data(self, pickle):
        s = Scalar(name='s')
        a = Scalar(name='a')

        sdata = SharedData(name='sdata', npthreads=2, cfields=[s], ncfields=[a])

        pkl_sdata = pickle.dumps(sdata)
        new_sdata = pickle.loads(pkl_sdata)

        assert sdata.name == new_sdata.name
        assert sdata.shape == new_sdata.shape
        assert sdata.size == new_sdata.size
        assert sdata.fields == new_sdata.fields
        assert sdata.pfields == new_sdata.pfields
        assert sdata.cfields == new_sdata.cfields
        assert sdata.ncfields == new_sdata.ncfields

        ffp = FieldFromPointer(sdata.symbolic_flag, sdata.indexed)

        pkl_ffp = pickle.dumps(ffp)
        new_ffp = pickle.loads(pkl_ffp)

        assert ffp.field == new_ffp.field
        assert ffp.base.name == new_ffp.base.name
        assert ffp.function.fields == new_ffp.function.fields

        indexed = sdata[0]

        pkl_indexed = pickle.dumps(indexed)
        new_indexed = pickle.loads(pkl_indexed)

        assert indexed.name == new_indexed.name

    def test_findexed(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid)

        strides_map = {x: 1, y: 2, z: 3}
        fi = FIndexed(f.base, x+1, y, z-2, strides_map=strides_map, accessor='fL')

        pkl_fi = pickle.dumps(fi)
        new_fi = pickle.loads(pkl_fi)

        assert new_fi.name == fi.name
        assert new_fi.accessor == 'fL'
        assert new_fi.indices == (x+1, y, z-2)
        assert new_fi.strides_map == fi.strides_map

    def test_component_access(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid)

        ca = ComponentAccess(f.indexify(), 1)

        pkl_ca = pickle.dumps(ca)
        new_ca = pickle.loads(pkl_ca)

        assert new_ca.index == 1
        assert new_ca.function.name == f.name

    def test_symbolics(self, pickle):
        a = Symbol('a')

        id = IntDiv(a, 3)
        pkl_id = pickle.dumps(id)
        new_id = pickle.loads(pkl_id)
        assert id == new_id

        ffp = CallFromPointer('foo', a, ['b', 'c'])
        pkl_ffp = pickle.dumps(ffp)
        new_ffp = pickle.loads(pkl_ffp)
        assert ffp == new_ffp

        li = ListInitializer(['a', 'b'])
        pkl_li = pickle.dumps(li)
        new_li = pickle.loads(pkl_li)
        assert li == new_li

        df = DefFunction('f', ['a', 1, 2])
        pkl_df = pickle.dumps(df)
        new_df = pickle.loads(pkl_df)
        assert df == new_df
        assert df.arguments == new_df.arguments

    def test_timers(self, pickle):
        """Pickling for Timers used in Operators for C-level profiling."""
        timer = Timer('timer', ['sec0', 'sec1'])
        pkl_obj = pickle.dumps(timer)
        new_obj = pickle.loads(pkl_obj)
        assert new_obj.name == timer.name
        assert new_obj.sections == timer.sections
        assert new_obj.value._obj.sec0 == timer.value._obj.sec0 == 0.0
        assert new_obj.value._obj.sec1 == timer.value._obj.sec1 == 0.0

    def test_guard_factor(self, pickle):
        d = Dimension(name='d')
        cd = ConditionalDimension(name='cd', parent=d, factor=4)

        gf = GuardFactor(cd)

        pkl_gf = pickle.dumps(gf)
        new_gf = pickle.loads(pkl_gf)

        assert str(gf) == str(new_gf)

    def test_guard_bound(self, pickle):
        d = Dimension(name='d')

        gb = GuardBound(d, 3)

        pkl_gb = pickle.dumps(gb)
        new_gb = pickle.loads(pkl_gb)

        assert str(gb) == str(new_gb)

    @pytest.mark.parametrize('direction', [Backward, Forward])
    def test_guard_bound_next(self, pickle, direction):
        d = Dimension(name='d')
        cd = ConditionalDimension(name='cd', parent=d, factor=4)

        for i in [d, cd]:
            gbn = GuardBoundNext(i, direction)

            pkl_gbn = pickle.dumps(gbn)
            new_gbn = pickle.loads(pkl_gbn)

            assert str(gbn) == str(new_gbn)

    def test_temp_function(self, pickle):
        grid = Grid(shape=(3, 3))
        d = Dimension(name='d')

        cf = TempFunction(name='f', dtype=np.float64, dimensions=grid.dimensions,
                          halo=((1, 1), (1, 1)))

        pkl_cf = pickle.dumps(cf)
        new_cf = pickle.loads(pkl_cf)
        assert new_cf.name == cf.name
        assert new_cf.dtype is np.float64
        assert new_cf.halo == ((1, 1), (1, 1))
        assert new_cf.ndim == cf.ndim
        assert new_cf.dim is None

        pcf = cf._make_pointer(d)

        pkl_pcf = pickle.dumps(pcf)
        new_pcf = pickle.loads(pkl_pcf)
        assert new_pcf.name == pcf.name
        assert new_pcf.dim.name == 'd'
        assert new_pcf.ndim == cf.ndim + 1
        assert new_pcf.halo == ((0, 0), (1, 1), (1, 1))

    def test_deviceid(self, pickle):
        did = DeviceID()

        pkl_did = pickle.dumps(did)
        new_did = pickle.loads(pkl_did)
        # TODO: this will be extend when we'll support DeviceID
        # for multi-node multi-gpu execution, when DeviceID will have
        # to pick its default value from an MPI communicator attached
        # to the runtime arguments

        assert did.name == new_did.name
        assert did.dtype == new_did.dtype

    def test_npthreads(self, pickle):
        npt = NPThreads(name='npt', size=3)

        pkl_npt = pickle.dumps(npt)
        new_npt = pickle.loads(pkl_npt)

        assert npt.name == new_npt.name
        assert npt.dtype == new_npt.dtype
        assert npt.size == new_npt.size

    def test_receiver(self, pickle):
        grid = Grid(shape=(3,))
        time_range = TimeAxis(start=0., stop=1000., step=0.1)
        nreceivers = 3

        rec = Receiver(name='rec', grid=grid, time_range=time_range, npoint=nreceivers,
                       coordinates=[(0.,), (1.,), (2.,)])
        rec.data[:] = 1.

        pkl_rec = pickle.dumps(rec)
        new_rec = pickle.loads(pkl_rec)

        assert np.all(new_rec.data == 1)
        assert np.all(new_rec.coordinates.data == [[0.], [1.], [2.]])

    @pytest.mark.parametrize('transpose', [direct, transpose])
    @pytest.mark.parametrize('side', [left, right, centered])
    @pytest.mark.parametrize('deriv_order', [1, 2])
    @pytest.mark.parametrize('fd_order', [2, 4])
    @pytest.mark.parametrize('x0', ["{}", "{x: x + x.spacing/2}"])
    @pytest.mark.parametrize('method', ['FD', 'RSFD'])
    @pytest.mark.parametrize('weights', [None, [1., 2., 3.]])
    def test_derivative(self, pickle, transpose, side, deriv_order,
                        fd_order, x0, method, weights):
        grid = Grid(shape=(3,))
        x = grid.dimensions[0]
        x0 = eval(x0)
        f = Function(name='f', grid=grid, space_order=4)
        dfdx = f.diff(x, order=deriv_order, fd_order=fd_order, side=side,
                      x0=x0, method=method, weights=weights)

        pkl_dfdx = pickle.dumps(dfdx)
        new_dfdx = pickle.loads(pkl_dfdx)

        assert new_dfdx.dims == dfdx.dims
        assert new_dfdx.side == dfdx.side
        assert new_dfdx.fd_order == dfdx.fd_order
        assert new_dfdx.deriv_order == dfdx.deriv_order
        assert new_dfdx.x0 == dfdx.x0
        assert new_dfdx.method == dfdx.method
        assert new_dfdx.weights == dfdx.weights

    def test_equation(self, pickle):
        grid = Grid(shape=(3,))
        x = grid.dimensions[0]
        f = Function(name='f', grid=grid)

        # Some implicit dim
        xs = ConditionalDimension(name='xs', parent=x, factor=4)

        eq = Eq(f, f+1, implicit_dims=xs)

        pkl_eq = pickle.dumps(eq)
        new_eq = pickle.loads(pkl_eq)

        assert new_eq.lhs.name == f.name
        assert str(new_eq.rhs) == 'f(x) + 1'
        assert new_eq.implicit_dims[0].name == 'xs'
        assert new_eq.implicit_dims[0].factor == 4

    @pytest.mark.parametrize('typ', [ctypes.c_float, 'struct truct'])
    def test_Cast(self, pickle, typ):
        a = Symbol('a')
        un = Cast(a, dtype=typ)

        pkl_un = pickle.dumps(un)
        new_un = pickle.loads(pkl_un)

        assert un == new_un

    @pytest.mark.parametrize('typ', [ctypes.c_float, 'struct truct'])
    def test_SizeOf(self, pickle, typ):
        un = SizeOf(typ)

        pkl_un = pickle.dumps(un)
        new_un = pickle.loads(pkl_un)

        assert un == new_un

    def test_pow_to_mul(self, pickle):
        grid = Grid(shape=(3,))
        f = Function(name='f', grid=grid)
        expr = pow_to_mul(f ** 2)

        assert expr.is_Mul

        pkl_expr = pickle.dumps(expr)
        new_expr = pickle.loads(pkl_expr)

        assert new_expr.is_Mul


class TestAdvanced:

    def test_foreign(self):
        MySparseFunction = type('MySparseFunction', (SparseFunction,), {'attr': 42})

        grid = Grid(shape=(3,))

        msf = MySparseFunction(name='msf', grid=grid, npoint=3, space_order=2,
                               coordinates=[(0.,), (1.,), (2.,)])

        # Plain `pickle` doesn't support pickling of dynamic classes
        with pytest.raises(Exception):
            pickle0.dumps(msf)

        # But `cloudpickle` does
        pkl_msf = pickle1.dumps(msf)
        new_msf = pickle1.loads(pkl_msf)

        assert new_msf.attr == 42
        assert new_msf.name == 'msf'
        assert new_msf.npoint == 3


@pytest.mark.parametrize('pickle', [pickle0, pickle1])
class TestOperator:

    def test_geometry(self, pickle):

        shape = (50, 50, 50)
        spacing = [10. for _ in shape]
        nbl = 10
        nrec = 10
        tn = 150.

        # Create two-layer model from preset
        model = demo_model(preset='layers-isotropic', vp_top=1., vp_bottom=2.,
                           spacing=spacing, shape=shape, nbl=nbl)
        # Source and receiver geometries
        src_coordinates = np.empty((1, len(spacing)))
        src_coordinates[0, :] = np.array(model.domain_size) * .5
        if len(shape) > 1:
            src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

        rec_coordinates = np.empty((nrec, len(spacing)))
        rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
        if len(shape) > 1:
            rec_coordinates[:, 1] = np.array(model.domain_size)[1] * .5
            rec_coordinates[:, -1] = model.origin[-1] + 2 * spacing[-1]
        geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                       t0=0.0, tn=tn, src_type='Ricker', f0=0.010)

        pkl_geom = pickle.dumps(geometry)
        new_geom = pickle.loads(pkl_geom)

        assert np.all(new_geom.src_positions == geometry.src_positions)
        assert np.all(new_geom.rec_positions == geometry.rec_positions)
        assert new_geom.f0 == geometry.f0
        assert np.all(new_geom.src_type == geometry.src_type)
        assert np.all(new_geom.src.data == geometry.src.data)
        assert new_geom.t0 == geometry.t0
        assert new_geom.tn == geometry.tn

    def test_operator_parameters(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        f = Function(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)
        h = TimeFunction(name='h', grid=grid, save=10)
        op = Operator(Eq(h.forward, h + g + f + 1))
        for i in op.parameters:
            pkl_i = pickle.dumps(i)
            pickle.loads(pkl_i)

    def test_unjitted_operator(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        f = Function(name='f', grid=grid)

        op = Operator(Eq(f, f + 1))

        pkl_op = pickle.dumps(op)
        new_op = pickle.loads(pkl_op)

        assert str(op) == str(new_op)

    def test_operator_function(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        f = Function(name='f', grid=grid)

        op = Operator(Eq(f, f + 1))
        op.apply()

        pkl_op = pickle.dumps(op)
        new_op = pickle.loads(pkl_op)

        assert str(op) == str(new_op)

        new_op.apply(f=f)
        assert np.all(f.data == 2)

    def test_operator_function_w_preallocation(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        f = Function(name='f', grid=grid)
        f.data

        op = Operator(Eq(f, f + 1))
        op.apply()

        pkl_op = pickle.dumps(op)
        new_op = pickle.loads(pkl_op)

        assert str(op) == str(new_op)

        new_op.apply(f=f)
        assert np.all(f.data == 2)

    def test_operator_timefunction(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        f = TimeFunction(name='f', grid=grid, save=3)

        op = Operator(Eq(f.forward, f + 1))
        op.apply(time=0)

        pkl_op = pickle.dumps(op)
        new_op = pickle.loads(pkl_op)

        assert str(op) == str(new_op)

        new_op.apply(time_m=1, time_M=1, f=f)
        assert np.all(f.data[2] == 2)

    def test_operator_timefunction_w_preallocation(self, pickle):
        grid = Grid(shape=(3, 3, 3))
        f = TimeFunction(name='f', grid=grid, save=3)
        f.data

        op = Operator(Eq(f.forward, f + 1))
        op.apply(time=0)

        pkl_op = pickle.dumps(op)
        new_op = pickle.loads(pkl_op)

        assert str(op) == str(new_op)

        new_op.apply(time_m=1, time_M=1, f=f)
        assert np.all(f.data[2] == 2)

    def test_collected_coeffs(self, pickle):
        grid = Grid(shape=(8, 8, 8))
        f = TimeFunction(name='f', grid=grid, space_order=4)

        op = Operator(Eq(f.forward, f.dx2 + 1))

        pkl_op = pickle.dumps(op)
        new_op = pickle.loads(pkl_op)

        assert str(op) == str(new_op)

    def test_elemental(self, pickle):
        """
        Tests that elemental functions don't get reconstructed differently.
        """
        grid = Grid(shape=(101, 101))
        time_range = TimeAxis(start=0.0, stop=1000.0, num=12)

        nrec = 101
        rec = Receiver(name='rec', grid=grid, npoint=nrec, time_range=time_range)

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)
        rec_term = rec.interpolate(expr=u)

        eq = rec_term.evaluate[2]
        eq = eq.func(eq.lhs, eq.rhs.args[0])

        op = Operator(eq)

        pkl_op = pickle.dumps(op)
        new_op = pickle.loads(pkl_op)

        op.cfunction
        new_op.cfunction

        assert str(op) == str(new_op)

    @pytest.mark.parallel(mode=[1])
    def test_mpi_objects(self, pickle, mode):
        grid = Grid(shape=(4, 4, 4))

        # Neighbours
        obj = grid.distributor._obj_neighborhood
        pkl_obj = pickle.dumps(obj)
        new_obj = pickle.loads(pkl_obj)
        assert obj.name == new_obj.name
        assert obj.pname == new_obj.pname
        assert obj.pfields == new_obj.pfields

        # Communicator
        obj = grid.distributor._obj_comm
        pkl_obj = pickle.dumps(obj)
        new_obj = pickle.loads(pkl_obj)
        assert obj.name == new_obj.name
        assert obj.dtype == new_obj.dtype

        # Status
        obj = MPIStatusObject(name='status')
        pkl_obj = pickle.dumps(obj)
        new_obj = pickle.loads(pkl_obj)
        assert obj.name == new_obj.name
        assert obj.dtype == new_obj.dtype

        # Request
        obj = MPIRequestObject(name='request')
        pkl_obj = pickle.dumps(obj)
        new_obj = pickle.loads(pkl_obj)
        assert obj.name == new_obj.name
        assert obj.dtype == new_obj.dtype

    def test_threadid(self, pickle):
        grid = Grid(shape=(4, 4, 4))
        f = TimeFunction(name='f', grid=grid)
        op = Operator(Eq(f.forward, f + 1.), opt=('advanced', {'openmp': True}))

        tid = ThreadID(op.nthreads)

        pkl_tid = pickle.dumps(tid)
        new_tid = pickle.loads(pkl_tid)

        assert tid.name == new_tid.name
        assert tid.nthreads.name == new_tid.nthreads.name
        assert tid.symbolic_min.name == new_tid.symbolic_min.name
        assert tid.symbolic_max.name == new_tid.symbolic_max.name

    @pytest.mark.parallel(mode=[2])
    def test_mpi_grid(self, pickle, mode):
        grid = Grid(shape=(4, 4, 4))

        pkl_grid = pickle.dumps(grid)
        new_grid = pickle.loads(pkl_grid)

        assert grid.distributor.comm.size == 2
        assert new_grid.distributor.comm.size == 1  # Using cloned MPI_COMM_SELF
        assert grid.distributor.shape == (2, 4, 4)
        assert new_grid.distributor.shape == (4, 4, 4)

        # Same as before but only one rank calls `loads`. We make sure this
        # won't cause any hanging (this was an issue in the past when we're
        # using MPI_COMM_WORLD at unpickling
        if MPI.COMM_WORLD.rank == 1:
            new_grid = pickle.loads(pkl_grid)
            assert new_grid.distributor.comm.size == 1
        MPI.COMM_WORLD.Barrier()

    @pytest.mark.parallel(mode=[(1, 'full')])
    def test_mpi_fullmode_objects(self, pickle, mode):
        grid = Grid(shape=(4, 4, 4))
        x, y, _ = grid.dimensions

        # Message
        f = Function(name='f', grid=grid)
        obj = MPIMsgEnriched('msg', f, [Halo(x, LEFT)])
        pkl_obj = pickle.dumps(obj)
        new_obj = pickle.loads(pkl_obj)
        assert obj.name == new_obj.name
        assert obj.target.name == new_obj.target.name
        assert all(obj.target.dimensions[i].name == new_obj.target.dimensions[i].name
                   for i in range(grid.dim))
        assert new_obj.target.dimensions[0] is new_obj.halos[0].dim

        # Region
        x_m, x_M = x.symbolic_min, x.symbolic_max
        y_m, y_M = y.symbolic_min, y.symbolic_max
        obj = MPIRegion('reg', 1, [y, x],
                        [(((x, OWNED, LEFT),), {x: (x_m, Min(x_M, x_m))}),
                         (((y, OWNED, LEFT),), {y: (y_m, Min(y_M, y_m))})])
        pkl_obj = pickle.dumps(obj)
        new_obj = pickle.loads(pkl_obj)
        assert obj.prefix == new_obj.prefix
        assert obj.key == new_obj.key
        assert obj.name == new_obj.name
        assert len(new_obj.arguments) == 2
        assert all(d0.name == d1.name for d0, d1 in zip(obj.arguments, new_obj.arguments))
        assert all(new_obj.arguments[i] is new_obj.owned[i][0][0][0]  # `x` and `y`
                   for i in range(2))
        assert new_obj.owned[0][0][0][1] is new_obj.owned[1][0][0][1]  # `OWNED`
        assert new_obj.owned[0][0][0][2] is new_obj.owned[1][0][0][2]  # `LEFT`
        for n, i in enumerate(new_obj.owned):
            d, v = list(i[1].items())[0]
            assert d is new_obj.arguments[n]
            assert v[0] is d.symbolic_min
            assert v[1] == Min(d.symbolic_max, d.symbolic_min)

    @pytest.mark.parallel(mode=[(1, 'basic'), (1, 'full')])
    def test_mpi_operator(self, pickle, mode):
        grid = Grid(shape=(4,))
        f = TimeFunction(name='f', grid=grid)

        # Using `sum` creates a stencil in `x`, which in turn will
        # trigger the generation of code for MPI halo exchange
        op = Operator(Eq(f.forward, f.sum() + 1))
        op.apply(time=2)

        pkl_op = pickle.dumps(op)
        new_op = pickle.loads(pkl_op)

        assert str(op) == str(new_op)

        new_grid = new_op.input[0].grid
        g = TimeFunction(name='g', grid=new_grid)
        new_op.apply(time=2, f=g)
        assert np.all(f.data[0] == [2., 3., 3., 3.])
        assert np.all(f.data[1] == [3., 6., 7., 7.])
        assert np.all(g.data[0] == f.data[0])
        assert np.all(g.data[1] == f.data[1])

    def test_full_model(self, pickle):
        shape = (50, 50, 50)
        spacing = [10. for _ in shape]
        nbl = 10

        # Create two-layer model from preset
        model = demo_model(preset='layers-isotropic', vp_top=1., vp_bottom=2.,
                           spacing=spacing, shape=shape, nbl=nbl)

        # Test Model pickling
        pkl_model = pickle.dumps(model)
        new_model = pickle.loads(pkl_model)
        assert np.isclose(np.linalg.norm(model.vp.data[:]-new_model.vp.data[:]), 0)

        f0 = .010
        dt = model.critical_dt
        t0 = 0.0
        tn = 350.0
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        # Test TimeAxis pickling
        pkl_time_range = pickle.dumps(time_range)
        new_time_range = pickle.loads(pkl_time_range)
        assert np.isclose(np.linalg.norm(time_range.time_values),
                          np.linalg.norm(new_time_range.time_values))

        # Test Class Constant pickling
        pkl_origin = pickle.dumps(model.grid.origin_symbols)
        new_origin = pickle.loads(pkl_origin)

        for a, b in zip(model.grid.origin_symbols, new_origin):
            assert a.compare(b) == 0

        # Test Class TimeDimension pickling
        time_dim = TimeDimension(name='time',
                                 spacing=Constant(name='dt', dtype=np.float32))
        pkl_time_dim = pickle.dumps(time_dim)
        new_time_dim = pickle.loads(pkl_time_dim)
        assert time_dim.spacing._value == new_time_dim.spacing._value

        # Test Class SteppingDimension
        stepping_dim = SteppingDimension(name='t', parent=time_dim)
        pkl_stepping_dim = pickle.dumps(stepping_dim)
        new_stepping_dim = pickle.loads(pkl_stepping_dim)
        assert stepping_dim.is_Time == new_stepping_dim.is_Time

        # Test Grid pickling
        pkl_grid = pickle.dumps(model.grid)
        new_grid = pickle.loads(pkl_grid)
        assert model.grid.shape == new_grid.shape

        assert model.grid.extent == new_grid.extent
        assert model.grid.shape == new_grid.shape
        for a, b in zip(model.grid.dimensions, new_grid.dimensions):
            assert a.compare(b) == 0

        ricker = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range)

        pkl_ricker = pickle.dumps(ricker)
        new_ricker = pickle.loads(pkl_ricker)
        assert np.isclose(np.linalg.norm(ricker.data), np.linalg.norm(new_ricker.data))
        # FIXME: fails randomly when using data.flatten() AND numpy is using MKL

    @pytest.mark.parametrize('subs', [False, True])
    def test_usave_sampled(self, pickle, subs):
        grid = Grid(shape=(10, 10, 10))
        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=8)

        time_range = TimeAxis(start=0, stop=1000, step=1)

        factor = Constant(name="factor", value=10, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=grid.time_dim,
                                        factor=factor)

        u0_save = TimeFunction(name="u0_save", grid=grid, time_dim=time_sub)
        src = RickerSource(name="src", grid=grid, time_range=time_range, f0=10)

        pde = u.dt2 - u.laplace
        stencil = Eq(u.forward, solve(pde, u.forward))

        src_term = src.inject(field=u.forward, expr=src)

        eqn = [stencil] + src_term
        eqn += [Eq(u0_save, u)]

        subs = grid.spacing_map if subs else {}
        op_fwd = Operator(eqn, subs=subs)

        tmp_pickle_op_fn = "tmp_operator.pickle"
        pickle.dump(op_fwd, open(tmp_pickle_op_fn, "wb"))
        op_new = pickle.load(open(tmp_pickle_op_fn, "rb"))

        assert str(op_fwd) == str(op_new)
