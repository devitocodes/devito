"""
Tests of the seismic models, and of the acoustic and TTI solvers with a free
surface, on a Grid and on a CurvilinearGrid.
"""

import numpy as np
import pytest

from devito import Eq, Function, Grid, MeshMap, Operator
from examples.seismic import AcquisitionGeometry, Receiver, SeismicModel
from examples.seismic.free_surface import fold, mirror
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.tti import AnisotropicWaveSolver

X = 10.*np.arange(51)
TTI = {'epsilon': .1, 'delta': .05, 'theta': .3}
GRADIENT = np.linspace(1.5, 3., 41)[None, :]*np.ones((51, 1))


def model(vp=2., **kwargs):
    """A SeismicModel with a free surface and velocity `vp`."""
    return SeismicModel(origin=(0., 0.), spacing=(10., 10.), shape=(51, 41),
                        space_order=8, vp=vp, nbl=10, fs=True, **kwargs)


def hill(height=60.):
    """The depth of a Gaussian hill `height` m high, positive down."""
    return -height*np.exp(-((X - 250.)/80.)**2)


def shot(seismic_model, solver=AcousticWaveSolver, **kwargs):
    """Return the record of a shot, gathered on the first rank, None on the others."""
    geometry = AcquisitionGeometry(seismic_model,
                                   np.stack([X, np.full_like(X, 20.)], axis=1),
                                   np.array([[250., 40.]]), 0., 400., f0=0.015,
                                   src_type='Ricker')
    wave_solver = solver(seismic_model, geometry, space_order=8)
    record = wave_solver.forward(**kwargs)[0].data_gather()
    return None if record is None else np.asarray(record)


def geometry_of(seismic_model):
    """Return a geometry with a source 50 m deep and receivers 30 m deep."""
    return AcquisitionGeometry(seismic_model, np.stack([X, np.full_like(X, 30.)], axis=1),
                               np.array([[250., 50.]]), 0., 300., f0=0.015,
                               src_type='Ricker')


def random_data(seismic_model, geometry):
    """A Receiver of random data at the receivers of `geometry`."""
    data = Receiver(name='data', grid=seismic_model.grid,
                    time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    data.data[:] = np.random.default_rng(20).standard_normal(data.data.shape)
    return data


def relative(lhs, rhs):
    """The relative difference of the two sides of a dot test."""
    return abs(lhs - rhs)/abs(lhs)


def dot_test(seismic_model, solver):
    """
    Return the relative error of the dot test of `solver`, None off the first
    rank.
    """
    geometry = geometry_of(seismic_model)
    wave_solver = solver(seismic_model, geometry, space_order=8)
    rec = wave_solver.forward()[0].data_gather()
    data = random_data(seismic_model, geometry)
    srca = wave_solver.adjoint(rec=data)[0].data_gather()
    source, data = geometry.src.data_gather(), data.data_gather()
    if rec is None:
        return None
    return relative(np.vdot(rec, data), np.vdot(source, srca))


class TestModel:

    def test_layout(self):
        """
        A free surface adds `space_order` rows above it, and no absorbing layer
        there.
        """
        seismic_model = model()
        grid = seismic_model.grid
        assert grid.shape == (71, 59)
        assert grid.origin[-1] == -80.
        rows = seismic_model.free_surface_rows
        assert rows.surface.start == 8 and rows.mirrored.start == 9
        assert rows.above.thickness == rows.mirrored.thickness == 8
        assert seismic_model.parameter_padsizes[-1] == (8, 10)
        # Above the surface, between the lateral layers, the mask is the same as
        # in the interior: there is no absorbing layer at the top
        mask = np.asarray(seismic_model.damp.data)
        assert np.all(mask[10:-10, :9] == mask[35, 30]) and mask[35, -1] < mask[35, 30]

    def test_flat(self):
        """A flat surface gives the records of the Cartesian model."""
        cartesian, flat = model(), model(surface=np.zeros(51))
        assert flat.grid.is_Curvilinear and not cartesian.grid.is_Curvilinear
        assert flat.grid.shape == cartesian.grid.shape
        depth = np.asarray(flat.grid.coordinates[-1].data)
        assert np.allclose(depth[0], -80. + 10.*np.arange(59), rtol=0, atol=1e-4)
        assert flat.critical_dt == cartesian.critical_dt
        assert np.array_equal(shot(flat), shot(cartesian))

    def test_shared_grid(self):
        """
        A model built on the grid of another model pads its parameters the same
        way.
        """
        reference = model(vp=np.full((51, 41), 2.), surface=hill())
        smooth = model(vp=np.full((51, 41), 1.5), grid=reference.grid)
        assert smooth.parameter_padsizes == reference.parameter_padsizes
        assert smooth.vp.shape == reference.vp.shape == reference.grid.shape
        assert np.all(np.asarray(smooth.vp.data) == 1.5)

    def test_surface(self):
        """
        The surface is the row below the mirror rows, extended flat across the
        layers.
        """
        depth = np.asarray(model(surface=hill()).grid.coordinates[-1].data)
        assert np.allclose(depth[:, 8], np.pad(hill(), 10, mode='edge'), atol=1e-4)

    def test_parameters_above(self):
        """The parameters are extended from the surface up into the mirror rows."""
        vp = np.linspace(1.5, 3., 41)[None, :]*np.ones((51, 1))
        values = np.asarray(model(surface=hill(), vp=vp).vp.data)
        assert np.allclose(values[:, :8], values[:, 8:9])

    def test_update(self):
        """An updated parameter is padded as when the model is built with it."""
        vp = np.linspace(1.5, 3., 41)[None, :]*np.ones((51, 1))
        built = model(surface=hill(), vp=vp)
        updated = model(surface=hill(), vp=np.ones((51, 41)))
        updated.update('vp', vp)
        assert np.array_equal(np.asarray(updated.vp.data), np.asarray(built.vp.data))

    def test_critical_dt(self):
        """The time step scales with the smallest cell, under a deep valley."""
        deep = model(surface=-hill(80.))
        assert deep.grid.min_spacing < 10.
        expected = model().critical_dt*deep.grid.min_spacing/10.
        assert np.isclose(deep.critical_dt, expected, rtol=1e-2)

    def test_depth(self):
        """The same model given its map or its node depths records the same shot."""
        reference = model(surface=hill(), dtype=np.float64)
        physical = (slice(10, -10), slice(8, -10))
        given = model(depth=np.asarray(reference.grid.coordinates[-1].data)[physical],
                      dtype=np.float64)
        mapped = model(mapping=reference.grid.mapping, dtype=np.float64)
        assert isinstance(given.grid.mapping, MeshMap)
        assert given.grid.shape == mapped.grid.shape == reference.grid.shape

        records = [shot(m, dt=reference.critical_dt) for m in (reference, given, mapped)]
        assert np.array_equal(records[2], records[0])
        assert np.linalg.norm(records[1] - records[0]) < 4e-6*np.linalg.norm(records[0])

    def test_invalid(self):
        """Inconsistent arguments are refused."""
        zs = np.meshgrid(X, 10.*np.arange(41), indexing='ij')[1]
        with pytest.raises(ValueError, match='surface and depth'):
            model(depth=zs, surface=np.zeros(51))
        with pytest.raises(ValueError, match='grid and surface'):
            model(surface=np.zeros(51), grid=model().grid)
        with pytest.raises(ValueError, match='fs=True'):
            SeismicModel(origin=(0., 0.), spacing=(10., 10.), shape=(51, 41),
                         space_order=8, vp=2., nbl=10, surface=np.zeros(51))
        with pytest.raises(ValueError, match='depth 0'):
            SeismicModel(origin=(0., 100.), spacing=(10., 10.), shape=(51, 41),
                         space_order=8, vp=2., nbl=10, fs=True, surface=np.zeros(51))
        with pytest.raises(ValueError, match='one value per lateral node'):
            model(surface=np.zeros(50))
        with pytest.raises(ValueError, match='expected the shape of the model'):
            model(depth=zs[1:])

    def test_given_grid(self):
        """A grid given with a free surface must have the mirror rows."""
        reference = model()
        assert model(grid=reference.grid).grid is reference.grid
        without_rows = Grid(shape=(71, 51), extent=(700., 500.), origin=(-100., 0.))
        with pytest.raises(ValueError, match='59 rows from depth -80'):
            model(grid=without_rows)
        with pytest.raises(ValueError, match='space order 4'):
            SeismicModel(origin=(0., 0.), spacing=(10., 10.), shape=(51, 41),
                         space_order=4, vp=2., nbl=10, fs=True, grid=reference.grid)


class TestFreeSurface:

    @pytest.mark.parametrize('odd, zero', [(True, None), (False, None), (False, True)])
    @pytest.mark.parametrize('staggered', [False, True])
    @pytest.mark.parametrize('surface', [None, 'hill'])
    def test_fold(self, odd, zero, staggered, surface):
        """The fold is the adjoint of the mirror."""
        seismic_model = model(dtype=np.float64,
                              surface=None if surface is None else hill())
        grid = seismic_model.grid
        z = grid.dimensions[-1]
        u, v = (Function(name=n, grid=grid, space_order=8,
                         staggered=z if staggered else None) for n in 'uv')
        rng = np.random.default_rng(1)
        u.data[:] = rng.standard_normal(u.shape)
        v.data[:] = rng.standard_normal(v.shape)
        before = np.array(u.data), np.array(v.data)
        Operator(mirror(seismic_model, u, odd=odd, zero=zero) +
                 fold(seismic_model, v, 1, odd=odd, zero=zero)).apply()

        location = {d: u.indices_ref[d] for d in grid.dimensions}
        weight = Function(name='weight', grid=grid, staggered=u.staggered)
        Operator(Eq(weight, grid.jacobian_at(location))).apply()
        jacobian = np.asarray(weight.data)
        forward = np.sum(jacobian*np.asarray(u.data)*before[1])
        adjoint = np.sum(jacobian*before[0]*np.asarray(v.data))
        assert np.isclose(forward, adjoint, rtol=1e-13, atol=0)

    @pytest.mark.parametrize('solver, parameters', [(AcousticWaveSolver, {}),
                                                    (AnisotropicWaveSolver, TTI)])
    @pytest.mark.parametrize('surface', [None, 'hill'])
    def test_mirror(self, solver, parameters, surface):
        """Above the surface, the field is the odd mirror of the field below."""
        seismic_model = model(surface=None if surface is None else hill(),
                              **parameters)
        geometry = AcquisitionGeometry(seismic_model, np.array([[250., 20.]]),
                                       np.array([[250., 40.]]), 0., 100., f0=0.015,
                                       src_type='Ricker')
        fields = solver(seismic_model, geometry, space_order=8).forward()[1:-1]
        s = seismic_model.mirror_rows
        last = (geometry.nt - 1) % fields[0].time_size
        for field in fields:
            data = np.asarray(field.data[last])
            below = data[:, s + 1:2*s + 1]
            assert np.abs(below).max() > 0
            assert np.all(data[:, s] == 0)
            assert np.all(data[:, :s] == -below[:, ::-1])

    @pytest.mark.parametrize('solver, parameters', [(AcousticWaveSolver, {}),
                                                    (AnisotropicWaveSolver, TTI)])
    @pytest.mark.parametrize('surface', [None, 'flat'])
    def test_adjoint(self, solver, parameters, surface):
        """The adjoints pass the dot test under a flat surface."""
        seismic_model = model(vp=GRADIENT, dtype=np.float64,
                              surface=None if surface is None else np.zeros(51),
                              **parameters)
        assert dot_test(seismic_model, solver) < 1e-12

    def test_adjoint_hill(self):
        """The acoustic adjoint under a hill passes the dot test to roundoff."""
        seismic_model = model(surface=hill(), dtype=np.float64)
        assert dot_test(seismic_model, AcousticWaveSolver) < 1e-12

    @pytest.mark.parametrize('surface', [None, 'flat', 'hill'])
    def test_born(self, surface):
        """The Born operator and the gradient pass the dot test to roundoff."""
        surface = {None: None, 'flat': np.zeros(51), 'hill': hill()}[surface]
        seismic_model = model(vp=GRADIENT, surface=surface, dtype=np.float64)
        geometry = geometry_of(seismic_model)
        wave_solver = AcousticWaveSolver(seismic_model, geometry, space_order=8)
        dm = Function(name='dm', grid=seismic_model.grid, space_order=0)
        dm.data[:] = np.random.default_rng(1).standard_normal(dm.shape)
        linear = wave_solver.jacobian(dm)[0]
        u0 = wave_solver.forward(save=True)[1]
        data = random_data(seismic_model, geometry)
        gradient = wave_solver.jacobian_adjoint(data, u0)[0]
        jacobian = seismic_model.grid.jacobian_at({})
        weight = np.asarray(jacobian.data) if jacobian.is_Function else float(jacobian)
        assert relative(np.vdot(linear.data, data.data),
                        np.sum(weight*np.asarray(gradient.data)*np.asarray(dm.data))) \
            < 1e-12


class TestMPI:

    @pytest.mark.parallel(mode=2)
    def test_free_surface(self, mode):
        """
        Decomposing along the vertical gives the same records as decomposing
        laterally.
        """
        records = []
        for topology in ((1, 2), (2, 1)):
            seismic_model = SeismicModel(origin=(0., 0.), spacing=(10., 10.),
                                         shape=(51, 41), space_order=8, vp=2., nbl=10,
                                         fs=True, surface=hill(), topology=topology)
            records.append(shot(seismic_model))
        if records[0] is not None:
            difference = np.linalg.norm(records[0] - records[1])
            assert difference < 2e-6*np.linalg.norm(records[1])

        seismic_model = SeismicModel(origin=(0., 0.), spacing=(10., 10.),
                                     shape=(51, 41), space_order=8, vp=GRADIENT, nbl=10,
                                     fs=True, surface=np.zeros(51), topology=(1, 2),
                                     dtype=np.float64)
        result = dot_test(seismic_model, AcousticWaveSolver)
        if result is not None:
            assert result < 1e-12
