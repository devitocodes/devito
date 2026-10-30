"""
The seismic models and the acoustic and TTI example solvers with a free surface,
on a Grid and on a CurvilinearGrid.
"""

import numpy as np
import pytest

from devito import Eq, Function, Grid, MeshMap, Operator
from examples.seismic import AcquisitionGeometry, Receiver, SeismicModel
from examples.seismic.free_surface import fold, mirror
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.tti import AnisotropicWaveSolver

# The lateral nodes of the physical domain, the anisotropy of the TTI cases, and a
# velocity increasing with depth, which the free surface sees through `m`
X = 10.*np.arange(51)
TTI = {'epsilon': .1, 'delta': .05, 'theta': .3}
GRADIENT = np.linspace(1.5, 3., 41)[None, :]*np.ones((51, 1))


def model(vp=2., **kwargs):
    """A SeismicModel with a free surface, of velocity `vp`."""
    return SeismicModel(origin=(0., 0.), spacing=(10., 10.), shape=(51, 41),
                        space_order=8, vp=vp, nbl=10, fs=True, **kwargs)


def hill(height=60.):
    """A Gaussian hill of `height` m, as a depth positive down."""
    return -height*np.exp(-((X - 250.)/80.)**2)


def shot(seismic_model, solver=AcousticWaveSolver, **kwargs):
    """
    The record of a shot 40 m deep, the receivers 20 m deep, gathered on the
    first rank under MPI, None on the others.
    """
    geometry = AcquisitionGeometry(seismic_model,
                                   np.stack([X, np.full_like(X, 20.)], axis=1),
                                   np.array([[250., 40.]]), 0., 400., f0=0.015,
                                   src_type='Ricker')
    wave_solver = solver(seismic_model, geometry, space_order=8)
    record = wave_solver.forward(**kwargs)[0].data_gather()
    return None if record is None else np.asarray(record)


def geometry_of(seismic_model):
    """A shot 50 m deep, the receivers 30 m deep."""
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
    The relative difference of the dot test of the adjoint of `solver`, on
    the first rank under MPI, None on the others.
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
        A free surface gives the grid `space_order` rows above the surface, no
        absorbing layer there, and the SubDomains of the updates and the mirror.
        """
        seismic_model = model()
        grid = seismic_model.grid
        assert grid.shape == (71, 59)
        assert grid.origin[-1] == -80.
        rows = seismic_model.free_surface_rows
        assert rows.surface.start == 8 and rows.mirrored.start == 9
        assert rows.above.thickness == rows.mirrored.thickness == 8
        assert seismic_model.parameter_padsizes[-1] == (8, 10)
        # No absorbing layer above the surface, inside the lateral ones: the
        # mask there is that of the interior, below it that of the layer
        mask = np.asarray(seismic_model.damp.data)
        assert np.all(mask[10:-10, :9] == mask[35, 30]) and mask[35, -1] < mask[35, 30]

    def test_flat(self):
        """
        A flat surface gives a CurvilinearGrid with the nodes, the time step and
        the records of the Cartesian model.
        """
        cartesian, flat = model(), model(surface=np.zeros(51))
        assert flat.grid.is_Curvilinear and not cartesian.grid.is_Curvilinear
        assert flat.grid.shape == cartesian.grid.shape
        depth = np.asarray(flat.grid.coordinates[-1].data)
        assert np.allclose(depth[0], -80. + 10.*np.arange(59), rtol=0, atol=1e-4)
        assert flat.critical_dt == cartesian.critical_dt
        assert np.array_equal(shot(flat), shot(cartesian))

    def test_shared_grid(self):
        """
        A model built on the grid of a model over a surface, e.g. a smooth model
        for imaging, pads its parameters with the same mirror rows.
        """
        reference = model(vp=np.full((51, 41), 2.), surface=hill())
        smooth = model(vp=np.full((51, 41), 1.5), grid=reference.grid)
        assert smooth.parameter_padsizes == reference.parameter_padsizes
        assert smooth.vp.shape == reference.vp.shape == reference.grid.shape
        assert np.all(np.asarray(smooth.vp.data) == 1.5)

    def test_surface(self):
        """The surface is below the mirror rows, continued flat across the layers."""
        depth = np.asarray(model(surface=hill()).grid.coordinates[-1].data)
        assert np.allclose(depth[:, 8], np.pad(hill(), 10, mode='edge'), atol=1e-4)

    def test_parameters_above(self):
        """The parameters are continued from the surface into the mirror rows."""
        vp = np.linspace(1.5, 3., 41)[None, :]*np.ones((51, 1))
        values = np.asarray(model(surface=hill(), vp=vp).vp.data)
        assert np.allclose(values[:, :8], values[:, 8:9])

    def test_update(self):
        """
        A parameter updated with an array over the physical domain is padded as
        it is when the model is built with it, into the mirror rows too.
        """
        vp = np.linspace(1.5, 3., 41)[None, :]*np.ones((51, 1))
        built = model(surface=hill(), vp=vp)
        updated = model(surface=hill(), vp=np.ones((51, 41)))
        updated.update('vp', vp)
        assert np.array_equal(np.asarray(updated.vp.data), np.asarray(built.vp.data))

    def test_critical_dt(self):
        """The time step follows the smallest cell, under the deepest surface."""
        deep = model(surface=-hill(80.))
        assert deep.grid.min_spacing < 10.
        expected = model().critical_dt*deep.grid.min_spacing/10.
        assert np.isclose(deep.critical_dt, expected, rtol=1e-2)

    def test_coordinates(self):
        """
        A model given the map of the grid of a model over a surface, `mapping=`,
        records the same shot, and one given the nodes of its physical domain,
        `coordinates=`, the same to 3.3e-6: the two differ in the absorbing
        layers only, where the surface map continues the surface flat and the
        mesh its grid lines straight.
        """
        reference = model(surface=hill(), dtype=np.float64)
        physical = (slice(10, -10), slice(8, -10))
        nodes = tuple(np.asarray(c.data)[physical] for c in reference.grid.coordinates)
        given = model(coordinates=nodes, dtype=np.float64)
        mapped = model(mapping=reference.grid.mapping, dtype=np.float64)
        assert isinstance(given.grid.mapping, MeshMap)
        assert given.grid.shape == mapped.grid.shape == reference.grid.shape

        records = [shot(m, dt=reference.critical_dt) for m in (reference, given, mapped)]
        assert np.array_equal(records[2], records[0])
        assert np.linalg.norm(records[1] - records[0]) < 4e-6*np.linalg.norm(records[0])

    def test_invalid(self):
        """
        A model takes one of a grid, a surface, node coordinates or a map, needs
        a free surface on a CurvilinearGrid, a surface at depth 0, and node
        coordinates of its shape.
        """
        xs, zs = np.meshgrid(X, 10.*np.arange(41), indexing='ij')
        with pytest.raises(ValueError, match='surface and coordinates'):
            model(coordinates=(xs, zs), surface=np.zeros(51))
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
            model(coordinates=(xs[1:], zs[1:]))

    def test_given_grid(self):
        """
        A grid given with a free surface needs its rows above the surface, e.g.
        that of a model of the same space order, and not the layout of a grid
        without them.
        """
        reference = model()
        assert model(grid=reference.grid).grid is reference.grid
        without_rows = Grid(shape=(71, 51), extent=(700., 500.), origin=(-100., 0.))
        with pytest.raises(ValueError, match='59 rows from the depth -80'):
            model(grid=without_rows)
        with pytest.raises(ValueError, match='space order 4'):
            SeismicModel(origin=(0., 0.), spacing=(10., 10.), shape=(51, 41),
                         space_order=4, vp=2., nbl=10, fs=True, grid=reference.grid)


class TestFreeSurface:

    @pytest.mark.parametrize('odd, zero', [(True, None), (False, None), (False, True)])
    @pytest.mark.parametrize('staggered', [False, True])
    @pytest.mark.parametrize('surface', [None, 'hill'])
    def test_fold(self, odd, zero, staggered, surface):
        """
        The fold is the transpose of the mirror in the inner product weighted by
        the Jacobian, for odd images and even ones, zeroed on the surface or
        not, of fields on the nodes and staggered along the vertical, on a Grid
        and under a hill.
        """
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
        """
        A source within the reach of the mirror is injected before it: after the
        last step, the source still on, the rows above the surface are exactly
        the odd mirror of the rows below, and the surface row is zero.
        """
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
        """
        The adjoints of the acoustic and TTI solvers, the fold of the mirror and
        their equations run backward, pass the dot test to roundoff under a flat
        surface, on a Grid and on a CurvilinearGrid, with a velocity varying
        with depth and the absorbing layers next to the mirror rows.
        """
        seismic_model = model(vp=GRADIENT, dtype=np.float64,
                              surface=None if surface is None else np.zeros(51),
                              **parameters)
        assert dot_test(seismic_model, solver) < 1e-12

    @pytest.mark.parametrize('solver, parameters, measured', [
        (AcousticWaveSolver, {}, 5.8e-2), (AnisotropicWaveSolver, TTI, 1.3e-2)])
    def test_adjoint_hill(self, solver, parameters, measured):
        """
        Under a 60 m hill, the adjoints run the forward equations backward, as on
        main, `u.laplace` included, whose discretization by the chain rule is
        symmetric up to its truncation error: the dot test reads 5.8e-2 for the
        acoustic solver and 1.3e-2 for the TTI one, the receivers 3 cells below
        the surface, where the curvature of the metric terms is largest. An
        exact adjoint writes its second derivatives with `.T`, exact
        transposes (see `tests/test_curvilinear.py`).
        """
        seismic_model = model(surface=hill(), dtype=np.float64, **parameters)
        assert dot_test(seismic_model, solver) < 1.2*measured

    @pytest.mark.parametrize('surface, tolerance', [(None, 1e-12), ('flat', 1e-12),
                                                    ('hill', 5e-3)])
    def test_born(self, surface, tolerance):
        """
        The Born modelling and its adjoint, the gradient, with the free surface,
        pass the dot test in the inner product weighted by the Jacobian: to
        roundoff under a flat surface, and to 3.7e-3 under the hill, the
        asymmetry of `u.laplace` there.
        """
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
        jacobian = seismic_model.grid.jacobian
        weight = np.asarray(jacobian.data) if jacobian.is_Function else float(jacobian)
        assert relative(np.vdot(linear.data, data.data),
                        np.sum(weight*np.asarray(gradient.data)*np.asarray(dm.data))) \
            < tolerance


class TestMPI:

    @pytest.mark.parallel(mode=2)
    def test_free_surface(self, mode):
        """
        With the vertical Dimension decomposed, the records are those of a
        lateral decomposition, which leaves the free surface on every rank, to
        float32 roundoff, 9.4e-7, and the adjoint passes the dot test.
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

    @pytest.mark.parallel(mode=4)
    def test_free_surface_reach(self, mode):
        """
        The free surface is refused, on every rank, if the ranks above the surface
        cannot read the rows below it that the mirror reads: here 29 rows over 4
        ranks, 7 rows each, when the mirror reads 16 rows deep.
        """
        seismic_model = SeismicModel(origin=(0., 0.), spacing=(10., 10.),
                                     shape=(51, 11), space_order=8, vp=2., nbl=10,
                                     fs=True, topology=(1, 4))
        with pytest.raises(ValueError, match='at least 17 rows'):
            shot(seismic_model)
