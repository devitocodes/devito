import pytest
import numpy as np
from sympy import Symbol
import cloudpickle as pickle

from conftest import skipif
from devito import (Constant, Eq, Function, TimeFunction, SparseFunction, Grid,
                    Dimension, SubDimension, ConditionalDimension, TimeDimension,
                    SteppingDimension, Operator)
from devito.mpi.routines import MPIStatusObject, MPIRequestObject
from devito.types import Symbol as dSymbol, Scalar
from devito.profiling import Timer
from devito.symbolics import IntDiv, ListInitializer, FunctionFromPointer
from examples.seismic import (demo_model, AcquisitionGeometry,
                              TimeAxis, RickerSource, Receiver)

pytestmark = skipif('ops')


def test_constant():
    c = Constant(name='c')
    assert c.data == 0.
    c.data = 1.

    pkl_c = pickle.dumps(c)
    new_c = pickle.loads(pkl_c)

    # .data is initialized, so it should have been pickled too
    assert np.all(c.data == 1.)
    assert np.all(new_c.data == 1.)


def test_dimension():
    d = Dimension(name='d')

    pkl_d = pickle.dumps(d)
    new_d = pickle.loads(pkl_d)

    assert d.name == new_d.name
    assert d.dtype == new_d.dtype


def test_function():
    grid = Grid(shape=(3, 3, 3))
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


def test_sparse_function():
    grid = Grid(shape=(3,))
    sf = SparseFunction(name='sf', grid=grid, npoint=3, space_order=2,
                        coordinates=[(0.,), (1.,), (2.,)])
    sf.data[0] = 1.

    pkl_sf = pickle.dumps(sf)
    new_sf = pickle.loads(pkl_sf)

    # .data is initialized, so it should have been pickled too
    assert np.all(sf.data[0] == 1.)
    assert np.all(new_sf.data[0] == 1.)

    # coordinates should also have been pickled
    assert np.all(sf.coordinates.data == new_sf.coordinates.data)

    assert sf.space_order == new_sf.space_order
    assert sf.dtype == new_sf.dtype
    assert sf.npoint == new_sf.npoint


def test_internal_symbols():
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

    s = Scalar(name='s', nonnegative=True)
    pkl_s = pickle.dumps(s)
    new_s = pickle.loads(pkl_s)
    assert new_s.name == s.name
    assert new_s.assumptions0['nonnegative'] is True


def test_sub_dimension():
    di = SubDimension.middle('di', Dimension(name='d'), 1, 1)

    pkl_di = pickle.dumps(di)
    new_di = pickle.loads(pkl_di)

    assert di.name == new_di.name
    assert di.dtype == new_di.dtype
    assert di.parent == new_di.parent
    assert di._thickness == new_di._thickness
    assert di._interval == new_di._interval


def test_conditional_dimension():
    d = Dimension(name='d')
    s = Scalar(name='s')
    cd = ConditionalDimension(name='ci', parent=d, factor=4, condition=s > 3)

    pkl_cd = pickle.dumps(cd)
    pkl_cd = pickle.loads(pkl_cd)

    assert cd.name == pkl_cd.name
    assert cd.parent == pkl_cd.parent
    assert cd.factor == pkl_cd.factor
    assert cd.condition == pkl_cd.condition


def test_receiver():
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


def test_geometry():

    shape = (50, 50, 50)
    spacing = [10. for _ in shape]
    nbpml = 10
    nrec = 10
    tn = 150.

    # Create two-layer model from preset
    model = demo_model(preset='layers-isotropic', vp_top=1., vp_bottom=2.,
                       spacing=spacing, shape=shape, nbpml=nbpml)
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


def test_symbolics():
    a = Symbol('a')

    id = IntDiv(a, 3)
    pkl_id = pickle.dumps(id)
    new_id = pickle.loads(pkl_id)
    assert id == new_id

    ffp = FunctionFromPointer('foo', a, ['b', 'c'])
    pkl_ffp = pickle.dumps(ffp)
    new_ffp = pickle.loads(pkl_ffp)
    assert ffp == new_ffp

    li = ListInitializer(['a', 'b'])
    pkl_li = pickle.dumps(li)
    new_li = pickle.loads(pkl_li)
    assert li == new_li


def test_timers():
    """Pickling for Timers used in Operators for C-level profiling."""
    timer = Timer('timer', ['sec0', 'sec1'])
    pkl_obj = pickle.dumps(timer)
    new_obj = pickle.loads(pkl_obj)
    assert new_obj.name == timer.name
    assert new_obj.sections == timer.sections
    assert new_obj.value._obj.sec0 == timer.value._obj.sec0 == 0.0
    assert new_obj.value._obj.sec1 == timer.value._obj.sec1 == 0.0


def test_operator_parameters():
    grid = Grid(shape=(3, 3, 3))
    f = Function(name='f', grid=grid)
    g = TimeFunction(name='g', grid=grid)
    h = TimeFunction(name='h', grid=grid, save=10)
    op = Operator(Eq(h.forward, h + g + f + 1))
    for i in op.parameters:
        pkl_i = pickle.dumps(i)
        pickle.loads(pkl_i)


def test_unjitted_operator():
    grid = Grid(shape=(3, 3, 3))
    f = Function(name='f', grid=grid)

    op = Operator(Eq(f, f + 1))

    pkl_op = pickle.dumps(op)
    new_op = pickle.loads(pkl_op)

    assert str(op) == str(new_op)


def test_operator_function():
    grid = Grid(shape=(3, 3, 3))
    f = Function(name='f', grid=grid)

    op = Operator(Eq(f, f + 1))
    op.apply()

    pkl_op = pickle.dumps(op)
    new_op = pickle.loads(pkl_op)

    assert str(op) == str(new_op)

    new_op.apply(f=f)
    assert np.all(f.data == 2)


def test_operator_function_w_preallocation():
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


def test_operator_timefunction():
    grid = Grid(shape=(3, 3, 3))
    f = TimeFunction(name='f', grid=grid, save=3)

    op = Operator(Eq(f.forward, f + 1))
    op.apply(time=0)

    pkl_op = pickle.dumps(op)
    new_op = pickle.loads(pkl_op)

    assert str(op) == str(new_op)

    new_op.apply(time_m=1, time_M=1, f=f)
    assert np.all(f.data[2] == 2)


def test_operator_timefunction_w_preallocation():
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


@skipif(['yask', 'nompi'])
@pytest.mark.parallel(mode=[1])
def test_mpi_objects():
    # Neighbours
    grid = Grid(shape=(4, 4, 4))
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


@skipif(['yask', 'nompi'])
@pytest.mark.parallel(mode=[1])
def test_mpi_operator():
    grid = Grid(shape=(4,))
    f = TimeFunction(name='f', grid=grid)
    g = TimeFunction(name='g', grid=grid)

    # Using `sum` creates a stencil in `x`, which in turn will
    # trigger the generation of code for MPI halo exchange
    op = Operator(Eq(f.forward, f.sum() + 1))
    op.apply(time=2)

    pkl_op = pickle.dumps(op)
    new_op = pickle.loads(pkl_op)

    assert str(op) == str(new_op)

    new_op.apply(time=2, f=g)
    assert np.all(f.data[0] == [2., 3., 3., 3.])
    assert np.all(f.data[1] == [3., 6., 7., 7.])
    assert np.all(g.data[0] == f.data[0])
    assert np.all(g.data[1] == f.data[1])


def test_full_model():

    shape = (50, 50, 50)
    spacing = [10. for _ in shape]
    nbpml = 10

    # Create two-layer model from preset
    model = demo_model(preset='layers-isotropic', vp_top=1., vp_bottom=2.,
                       spacing=spacing, shape=shape, nbpml=nbpml)

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
    pkl_origin = pickle.dumps(model.grid.origin)
    new_origin = pickle.loads(pkl_origin)

    for a, b in zip(model.grid.origin, new_origin):
        assert a.compare(b) == 0

    # Test Class TimeDimension pickling
    time_dim = TimeDimension(name='time', spacing=Constant(name='dt', dtype=np.float32))
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
